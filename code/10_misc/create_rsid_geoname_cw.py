"""Build crosswalks from RSIDs and IPEDS UNITIDs to Geonames city ids.

This script mirrors the legacy CBSA crosswalk workflow in
``create_rsid_cbsa_cw.py`` but replaces the HUD/CBSA chain with direct
matches against the Geonames ``cities.txt`` reference. The resulting files
live in ``{root}/data/int``:

* ``rsid_geoname_cw.parquet`` — Revelio RSIDs to ``geoname_id``
* ``ipeds_geoname_cw.parquet`` — IPEDS UNITIDs to ``geoname_id``
* ``rsid_ipeds_cw.parquet`` — Revelio RSIDs linked to IPEDS UNITIDs via shared geonames

The matching strategy is:

1. Normalize campus city / state pairs from FOIA and IPEDS sources.
2. Build an expanded Geonames lookup using the primary, ASCII, and alternate
   place names from ``cities.txt`` (falls back to ``cities500.txt`` when
   necessary).
3. Resolve each unique city/state pair via exact matches, then fuzzy
   (Levenshtein + Jaro-Winkler) within the same state when exact matches are
   unavailable.
4. Attach the best-scoring geoname id back to the FOIA/IPEDS tables and then
   to the RSID counts.
5. For RSIDs whose university country is missing or non-US, search for city
   tokens within their institution names using the global Geonames ``cities500``
   reference to recover additional matches.

Run the script with ``python 10_misc/create_rsid_geoname_cw.py``. Set the
``FROMSCRATCH`` flag below to rebuild intermediate FOIA aggregates if
necessary.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import duckdb as ddb
import numpy as np
import pandas as pd
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # sets cwd and shared paths
import helpers as help


FROMSCRATCH = False  # toggle to rebuild FOIA aggregates instead of reusing cached parquet

DATA_INT = Path(root) / "data" / "int"
DATA_RAW = Path(root) / "data" / "raw"
GEONAMES_DIR = Path(root) / "data" / "crosswalks" / "geonames"
GEONAMES_CANDIDATE_FILES = ["cities.txt", "cities500.txt"]

WRDS_USERS_PATH = DATA_INT / "wrds_users_sep2.parquet"
UNIV_ZIP_PATH = DATA_INT / "univ_zip_cw.parquet" # from FOIA data
RSID_OUTPUT_PATH = DATA_INT / "rsid_geoname_cw.parquet"
IPEDS_OUTPUT_PATH = DATA_INT / "ipeds_geoname_cw.parquet"
RSID_IPEDS_OUTPUT_PATH = DATA_INT / "rsid_ipeds_cw.parquet"
ISO_COUNTRY_CODES_PATH = Path(root) / "data" / "crosswalks" / "iso_country_codes.csv"

GEONAMES_COLUMNS = [
    "geoname_id",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]

NAME_CLEAN_RE = re.compile(r"[^a-z0-9 ]+")

if ISO_COUNTRY_CODES_PATH.exists():
    _iso_country_df = pd.read_csv(ISO_COUNTRY_CODES_PATH, usecols=["name", "alpha-2"])
    ISO_COUNTRY_MAP = {
        str(name).strip().lower(): str(code).strip().upper()
        for name, code in zip(_iso_country_df["name"], _iso_country_df["alpha-2"])
        if pd.notna(name) and pd.notna(code)
    }
else:
    ISO_COUNTRY_MAP: dict[str, str] = {}

STATE_NAME_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "washington dc": "DC",
    "dc": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "puerto rico": "PR",
    "guam": "GU",
    "american samoa": "AS",
    "northern mariana islands": "MP",
    "us virgin islands": "VI",
}


def _normalize_state_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    upper = text.upper()
    if len(upper) == 2 and upper.isalpha():
        return upper
    key = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    abbr = STATE_NAME_TO_ABBR.get(key)
    if abbr:
        return abbr
    key_nospace = key.replace(" ", "")
    for name, candidate in STATE_NAME_TO_ABBR.items():
        if name.replace(" ", "") == key_nospace:
            return candidate
    return upper if len(upper) == 2 and upper.isalpha() else ""


def _normalize(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(NAME_CLEAN_RE, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_geoname_variants() -> pd.DataFrame:
    """Return US geoname name variants suitable for city/state matching."""

    geoname_path: Optional[Path] = None
    for candidate in GEONAMES_CANDIDATE_FILES:
        path = GEONAMES_DIR / candidate
        if path.exists():
            geoname_path = path
            break
    if geoname_path is None:
        raise FileNotFoundError(
            f"Could not find any geonames cities file under {GEONAMES_DIR} (looked for: {GEONAMES_CANDIDATE_FILES})."
        )

    df = pd.read_csv(
        geoname_path,
        sep="\t",
        header=None,
        names=GEONAMES_COLUMNS,
        dtype={"geoname_id": str, "admin1_code": str, "alternatenames": str},
        na_values={"": np.nan, "\\N": np.nan},
        keep_default_na=True,
    )
    df = df[df["country_code"] == "US"].copy()
    df["admin1_code"] = df["admin1_code"].fillna("").str.upper()

    base_cols = ["geoname_id", "admin1_code", "latitude", "longitude", "population"]

    variants: list[pd.DataFrame] = []
    for col in ("name", "asciiname"):
        chunk = df[base_cols + [col]].rename(columns={col: "variant_raw"})
        variants.append(chunk)

    alt = df.loc[df["alternatenames"].notna(), base_cols + ["alternatenames"]].copy()
    if not alt.empty:
        alt["variant_raw"] = alt["alternatenames"].str.split(",")
        alt = alt.explode("variant_raw")
        variants.append(alt[base_cols + ["variant_raw"]])

    combined = pd.concat(variants, ignore_index=True)
    combined["variant_clean"] = _normalize(combined["variant_raw"])
    combined = combined[combined["variant_clean"].ne("")]
    combined = combined.drop_duplicates(subset=["geoname_id", "admin1_code", "variant_clean"])
    return combined


def resolve_city_state_pairs(pairs: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """Attach the best geoname match to each normalized city/state pair."""

    if pairs.empty:
        return pd.DataFrame(columns=[
            "city_clean",
            "state_clean",
            "geoname_id",
            "geo_name",
            "match_source",
            "match_score",
            "latitude",
            "longitude",
            "population",
        ])

    working = pairs.reset_index(drop=True).copy()
    working["pair_id"] = working.index

    exact = working.merge(
        variants,
        left_on=("city_clean", "state_clean"),
        right_on=("variant_clean", "admin1_code"),
        how="left",
    )
    exact = exact[exact["geoname_id"].notna()].copy()
    exact["match_source"] = "exact_city"
    exact["match_score"] = 1.0

    unmatched = working.loc[~working["pair_id"].isin(exact["pair_id"])]
    fuzzy = pd.DataFrame()
    if not unmatched.empty:
        left = unmatched[["pair_id", "city_clean", "state_clean"]].rename(columns={"state_clean": "state_code"})
        right = variants[[
            "geoname_id",
            "variant_clean",
            "variant_raw",
            "admin1_code",
            "latitude",
            "longitude",
            "population",
        ]].rename(columns={"admin1_code": "state_code"})
        matches = help.fuzzy_join_lev_jw(
            left,
            right,
            left_on="city_clean",
            right_on="variant_clean",
            block_key="state_code",
            threshold=0.9,
            top_n=3,
        )
        if not matches.empty:
            fuzzy = matches.rename(
                columns={
                    "pair_id_left": "pair_id",
                    "city_clean_left": "city_clean",
                    "state_code_left": "state_clean",
                    "geoname_id_right": "geoname_id",
                    "variant_clean_right": "variant_clean",
                    "variant_raw_right": "variant_raw",
                    "state_code_right": "admin1_code",
                }
            )
            fuzzy = fuzzy.assign(
                match_source="fuzzy_city",
                match_score=fuzzy.get("_score", np.nan),
            )

    candidates = pd.concat([exact, fuzzy], ignore_index=True, sort=False)
    if candidates.empty:
        best = working.copy()
        best[["geoname_id", "variant_raw", "match_source", "match_score", "latitude", "longitude", "population"]] = np.nan
    else:
        candidates["match_score"] = candidates["match_score"].fillna(0.0)
        candidates["population"] = candidates["population"].fillna(0.0)
        candidates["priority"] = candidates["match_source"].map({"exact_city": 0, "fuzzy_city": 1}).fillna(10)
        candidates = candidates.sort_values(
            ["pair_id", "priority", "match_score", "population"],
            ascending=[True, True, False, False],
        )
        best = candidates.groupby("pair_id", as_index=False).first()

    result = working.merge(best[[
        "pair_id",
        "geoname_id",
        "variant_raw",
        "match_source",
        "match_score",
        "latitude",
        "longitude",
        "population",
    ]], on="pair_id", how="left")
    result = result.drop(columns=["pair_id"])
    result = result.rename(columns={"variant_raw": "geo_name"})
    return result


def match_city_state(df: pd.DataFrame, city_col: str, state_col: str, variants: pd.DataFrame) -> pd.DataFrame:
    """Annotate ``df`` with normalized city/state and geoname ids."""

    data = df.copy()
    data["city_clean"] = _normalize(data[city_col])
    data["state_clean"] = data[state_col].apply(_normalize_state_value)

    mask = data["city_clean"].ne("") & data["state_clean"].ne("")
    valid_pairs = data.loc[mask, ["city_clean", "state_clean"]].drop_duplicates()
    resolved = resolve_city_state_pairs(valid_pairs, variants)

    merged = data.merge(
        resolved,
        on=["city_clean", "state_clean"],
        how="left",
    )
    return merged


def load_univ_zip(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    if UNIV_ZIP_PATH.exists() and not FROMSCRATCH:
        print("University Name to ZIP crosswalk already exists! Reading in...")
        return pd.read_parquet(UNIV_ZIP_PATH)

    f1_path = DATA_INT / "foia_sevp_combined_raw.parquet"
    query = f"""
        SELECT
            school_name,
            campus_zip_code,
            campus_state,
            campus_city,
            MIN(year) AS min_year,
            MAX(year) AS max_year
        FROM read_parquet('{f1_path}')
        GROUP BY 1,2,3,4
    """
    df = con.execute(query).df()
    df.to_parquet(UNIV_ZIP_PATH, index=False)
    return df


def load_univ_rsid(con: ddb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        SELECT rsid, university_raw, university_country, COUNT(*) AS n
        FROM read_parquet('{WRDS_USERS_PATH}')
        WHERE university_raw IS NOT NULL
          AND rsid IS NOT NULL
        GROUP BY rsid, university_raw, university_country
        HAVING COUNT(*) >= 10
    """
    return con.execute(query).df()


def load_ipeds_crosswalk() -> pd.DataFrame:
    ipeds_name_path = DATA_RAW / "ipeds_name_cw_2021.xlsx"
    ipeds_zip_path = DATA_RAW / "ipeds_cw_2021.csv"

    univ_cw = pd.read_excel(
        ipeds_name_path,
        sheet_name="Crosswalk",
        usecols=["OPEID", "IPEDSMatch", "PEPSSchname", "PEPSLocname", "IPEDSInstnm", "OPEIDMain", "IPEDSMain"],
    )
    univ_cw["UNITID"] = univ_cw["IPEDSMatch"].astype(str).str.replace("No match", "-1", regex=False).astype(int)

    zip_cw = pd.read_csv(
        ipeds_zip_path,
        usecols=["UNITID", "OPEID", "INSTNM", "CITY", "STABBR", "ZIP", "ALIAS"],
    )

    merged = (
        univ_cw[univ_cw["UNITID"] != -1]
        .merge(zip_cw, on=["UNITID", "OPEID"], how="left")
        .melt(
            id_vars=["UNITID", "CITY", "STABBR", "ZIP"],
            value_vars=["PEPSSchname", "PEPSLocname", "IPEDSInstnm", "INSTNM", "ALIAS"],
            var_name="source",
            value_name="instname",
        )
        .dropna(subset=["instname"])
        .drop_duplicates(subset=["UNITID", "instname"])
        .reset_index(drop=True)
    )
    merged["ZIP"] = merged.groupby("UNITID")["ZIP"].transform(lambda s: s.ffill().bfill()).astype(str).str.replace(r"-[0-9]+$", "", regex=True)
    merged["CITY"] = merged.groupby("UNITID")["CITY"].transform(lambda s: s.ffill().bfill())
    return merged


def load_global_city_tokens(min_token_length: int = 4) -> pd.DataFrame:
    """
    Build a lookup table of normalized city tokens from the Geonames cities500 file.

    Only non-US geonames are retained because the token matching is used for RSIDs
    whose university_country is missing or outside the United States.
    """

    primary_path = GEONAMES_DIR / "cities500.txt"
    fallback_path = GEONAMES_DIR / "cities.txt"
    if primary_path.exists():
        geoname_path = primary_path
    elif fallback_path.exists():
        geoname_path = fallback_path
    else:
        raise FileNotFoundError(
            f"Could not find cities500 or cities.txt under {GEONAMES_DIR}; token lookup unavailable."
        )

    df = pd.read_csv(
        geoname_path,
        sep="\t",
        header=None,
        names=GEONAMES_COLUMNS,
        dtype={"geoname_id": str, "admin1_code": str, "alternatenames": str, "country_code": str},
        na_values={"": np.nan, "\\N": np.nan},
        keep_default_na=True,
    )
    df["country_code"] = df["country_code"].str.upper()
    df = df[df["country_code"].ne("US")].copy()
    base_cols = ["geoname_id", "country_code", "population"]

    variants: list[pd.DataFrame] = []
    for col in ("name", "asciiname"):
        chunk = df[base_cols + [col]].rename(columns={col: "variant_raw"})
        variants.append(chunk)

    alt = df.loc[df["alternatenames"].notna(), base_cols + ["alternatenames"]].copy()
    if not alt.empty:
        alt["variant_raw"] = alt["alternatenames"].str.split(",")
        alt = alt.explode("variant_raw")
        variants.append(alt[base_cols + ["variant_raw"]])

    combined = pd.concat(variants, ignore_index=True)
    combined["variant_clean"] = _normalize(combined["variant_raw"])
    combined = combined[combined["variant_clean"].ne("")]
    combined["token"] = combined["variant_clean"].str.split(" ")
    combined = combined.explode("token")
    combined["token"] = combined["token"].str.strip()
    combined = combined[combined["token"].str.len() >= min_token_length]
    combined = combined.drop_duplicates(subset=["token", "geoname_id", "country_code"])
    combined = combined.rename(columns={"variant_raw": "geo_name"})
    return combined[["token", "geoname_id", "country_code", "geo_name", "population"]]


def build_school_geoname_lookup(school_geo: pd.DataFrame) -> pd.DataFrame:
    lookup = school_geo.dropna(subset=["geoname_id"]).copy()
    lookup["school_clean"] = _normalize(lookup["school_name"])
    lookup = lookup[lookup["school_clean"].ne("")]
    cols = [
        "school_name",
        "school_clean",
        "campus_city",
        "campus_state",
        "geoname_id",
        "geo_name",
        "match_source",
        "match_score",
    ]
    return lookup[cols].drop_duplicates(subset=["school_clean", "geoname_id"])


def map_rsid_to_geoname(
    univ_rsid: pd.DataFrame,
    school_lookup: pd.DataFrame,
    city_token_lookup: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    univ_rsid = univ_rsid.copy()
    univ_rsid["univ_clean"] = _normalize(univ_rsid["university_raw"])
    univ_rsid = univ_rsid[univ_rsid["univ_clean"].ne("")]
    if "university_country" in univ_rsid.columns:
        univ_rsid["country_clean"] = univ_rsid["university_country"].fillna("").astype(str).str.strip()
        univ_rsid["country_iso"] = univ_rsid["country_clean"].str.lower().map(ISO_COUNTRY_MAP)
    else:
        univ_rsid["country_clean"] = ""
        univ_rsid["country_iso"] = np.nan

    exact = univ_rsid.merge(
        school_lookup,
        left_on="univ_clean",
        right_on="school_clean",
        how="left",
    )
    exact_matches = exact[exact["geoname_id"].notna()].copy()
    exact_matches["match_method"] = exact_matches["match_source"].fillna("exact_name")
    exact_matches["match_score"] = exact_matches["match_score"].fillna(1.0)

    unmatched = univ_rsid.loc[~univ_rsid["rsid"].isin(exact_matches["rsid"])]
    fuzzy = pd.DataFrame()
    if not unmatched.empty:
        left = unmatched[["rsid", "university_raw", "univ_clean"]]
        right = school_lookup[[
            "school_name",
            "school_clean",
            "campus_city",
            "campus_state",
            "geoname_id",
            "geo_name",
            "match_source",
            "match_score",
        ]]
        matches = help.fuzzy_join_lev_jw(
            left,
            right,
            left_on="univ_clean",
            right_on="school_clean",
            threshold=0.9,
            top_n=3,
        )
        if not matches.empty:
            fuzzy = matches.rename(
                columns={
                    "rsid_left": "rsid",
                    "university_raw_left": "university_raw",
                    "univ_clean_left": "univ_clean",
                    "geoname_id_right": "geoname_id",
                    "geo_name_right": "geo_name",
                    "campus_city_right": "campus_city",
                    "campus_state_right": "campus_state",
                }
            )
            fuzzy = fuzzy.assign(
                match_method="fuzzy_name",
                match_score=fuzzy.get("_score", np.nan),
            )

    combined = pd.concat([exact_matches, fuzzy], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["geoname_id"])

    token_matches = pd.DataFrame()
    if (
        city_token_lookup is not None
        and not city_token_lookup.empty
    ):
        remaining = univ_rsid.loc[~univ_rsid["rsid"].isin(combined["rsid"])].copy()
        if not remaining.empty:
            target_mask = remaining["country_clean"].str.casefold().ne("united states")
            token_candidates = remaining.loc[target_mask & remaining["univ_clean"].ne("")]
            if not token_candidates.empty:
                token_candidates = token_candidates.assign(tokens=token_candidates["univ_clean"].str.split(" "))
                token_candidates = token_candidates.explode("tokens")
                token_candidates["token"] = token_candidates["tokens"].str.strip()
                token_candidates = token_candidates[token_candidates["token"].str.len() >= 4]
                token_candidates = token_candidates.drop_duplicates(subset=["rsid", "token"])
                token_matches = token_candidates.merge(city_token_lookup, on="token", how="inner")
                token_matches = token_matches[token_matches["geoname_id"].notna()].copy()
                if "country_iso" in token_matches.columns:
                    token_matches["country_code"] = token_matches["country_code"].str.upper()
                    token_matches = token_matches[
                        token_matches["country_iso"].isna()
                        | token_matches["country_iso"].eq(token_matches["country_code"])
                    ]
                if not token_matches.empty:
                    token_matches = token_matches.assign(
                        match_method="city_token",
                        match_score=token_matches["population"].fillna(0.0),
                        campus_city=pd.NA,
                        campus_state=pd.NA,
                    )
                    token_matches = token_matches.drop(columns=["tokens", "token"], errors="ignore")
                    combined = pd.concat([combined, token_matches], ignore_index=True, sort=False)

    combined = combined.dropna(subset=["geoname_id"])
    if combined.empty:
        return combined

    combined["match_score"] = combined["match_score"].fillna(0.0)
    combined["n"] = combined["n"].fillna(0.0)
    combined = combined.sort_values([
        "rsid",
        "match_score",
        "n",
    ], ascending=[True, False, False])
    combined = combined.drop_duplicates(subset=["rsid", "geoname_id"])
    combined["nmatch"] = combined.groupby("rsid")["geoname_id"].transform("nunique")
    return combined


def build_ipeds_geoname(ipeds_geo: pd.DataFrame) -> pd.DataFrame:
    df = ipeds_geo.dropna(subset=["geoname_id"]).copy()
    if df.empty:
        return df
    df = df.sort_values(["UNITID", "match_source", "match_score"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["UNITID", "geoname_id"])
    df["nmatch"] = df.groupby("UNITID")["geoname_id"].transform("nunique")
    keep_cols = [
        "UNITID",
        "instname",
        "CITY",
        "STABBR",
        "geoname_id",
        "geo_name",
        "match_source",
        "match_score",
        "nmatch",
    ]
    return df[keep_cols]


def build_rsid_ipeds_crosswalk(univ_rsid: pd.DataFrame, ipeds_names: pd.DataFrame) -> pd.DataFrame:
    """Link RSIDs directly to IPEDS UNITIDs via normalized university names (including aliases)."""

    columns = [
        "rsid",
        "university_raw",
        "unitid",
        "instname",
        "source",
        "match_count",
        "n",
    ]
    if univ_rsid.empty or ipeds_names.empty:
        return pd.DataFrame(columns=columns)

    rsid_df = univ_rsid.copy()
    rsid_df["name_clean"] = _normalize(rsid_df["university_raw"])
    rsid_df = rsid_df[rsid_df["name_clean"].ne("")]

    ipeds_df = ipeds_names.copy()
    ipeds_df["inst_clean"] = _normalize(ipeds_df["instname"])
    ipeds_df = ipeds_df[ipeds_df["inst_clean"].ne("")]

    if rsid_df.empty or ipeds_df.empty:
        return pd.DataFrame(columns=columns)

    source_priority = {
        "IPEDSInstnm": 0,
        "INSTNM": 1,
        "PEPSSchname": 2,
        "PEPSLocname": 3,
        "ALIAS": 4,
    }

    merged = rsid_df.merge(
        ipeds_df,
        left_on="name_clean",
        right_on="inst_clean",
        how="left",
        suffixes=("", "_ipeds"),
    )
    merged = merged[merged["UNITID"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["source_priority"] = merged["source"].map(source_priority).fillna(99)
    merged["match_count"] = merged.groupby("rsid")["UNITID"].transform("count")

    merged = merged.sort_values(
        ["rsid", "match_count", "source_priority", "UNITID", "instname"],
        ascending=[True, True, True, True, True],
    )
    merged = merged.drop_duplicates(subset=["rsid"])

    result = merged[
        [
            "rsid",
            "university_raw",
            "UNITID",
            "instname",
            "source",
            "match_count",
            "n",
        ]
    ].rename(columns={"UNITID": "unitid"})
    return result


def main() -> None:
    con = ddb.connect()

    geoname_variants = load_geoname_variants()
    geoname_sources = " and ".join(GEONAMES_CANDIDATE_FILES)
    print(f"Loaded {len(geoname_variants):,} geoname name variants for US cities from {geoname_sources}.")
    try:
        city_token_lookup = load_global_city_tokens()
        print(f"Loaded {len(city_token_lookup):,} global city tokens from Geonames for non-US RSID matching.")
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        city_token_lookup = pd.DataFrame()

    univ_zip_cw = load_univ_zip(con)
    univ_rsid_cw = load_univ_rsid(con)
    ipeds_cw = load_ipeds_crosswalk()

    print(f"FOIA schools: {len(univ_zip_cw):,} rows; RSID pairs: {len(univ_rsid_cw):,}; IPEDS names: {len(ipeds_cw):,}.")

    # matching FOIA school zips to geonames
    school_geo = match_city_state(univ_zip_cw, "campus_city", "campus_state", geoname_variants)
    resolved_school_share = school_geo["geoname_id"].notna().mean()
    print(f"Matched {resolved_school_share:.1%} of FOIA campus city/state pairs to geoname ids.")

    # matching RSIDs to geonames via school names and city tokens
    # TODO: FIX THIS!!!
    school_lookup = build_school_geoname_lookup(school_geo)
    rsid_geoname = map_rsid_to_geoname(univ_rsid_cw, school_lookup, city_token_lookup=city_token_lookup)
    if not rsid_geoname.empty:
        rsid_geoname.to_parquet(RSID_OUTPUT_PATH, index=False)
        print(
            f"Saved RSID to geoname crosswalk for {rsid_geoname['rsid'].nunique():,} RSIDs ({(rsid_geoname['nmatch']==1).sum():,} unique) to {RSID_OUTPUT_PATH}."
        )
    else:
        print("Warning: no RSID to geoname matches were produced.")

    ipeds_geo = match_city_state(ipeds_cw, "CITY", "STABBR", geoname_variants)
    ipeds_geoname = build_ipeds_geoname(ipeds_geo)
    if not ipeds_geoname.empty:
        ipeds_geoname.to_parquet(IPEDS_OUTPUT_PATH, index=False)
        print(
            f"Saved IPEDS to geoname crosswalk for {ipeds_geoname['UNITID'].nunique():,} UNITIDs to {IPEDS_OUTPUT_PATH}."
        )
    else:
        print("Warning: no IPEDS to geoname matches were produced.")

    rsid_ipeds = build_rsid_ipeds_crosswalk(univ_rsid_cw, ipeds_cw)
    if not rsid_ipeds.empty:
        rsid_ipeds.to_parquet(RSID_IPEDS_OUTPUT_PATH, index=False)
        print(
            f"Saved RSID to IPEDS crosswalk for {rsid_ipeds['rsid'].nunique():,} RSIDs ({rsid_ipeds['unitid'].nunique():,} UNITIDs) to {RSID_IPEDS_OUTPUT_PATH}."
        )
    else:
        print("Warning: no RSID to IPEDS matches were produced.")

    con.close()


if __name__ == "__main__":
    main()
