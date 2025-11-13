"""Build crosswalks from RSIDs and IPEDS UNITIDs to Geonames city ids.

This script mirrors the legacy CBSA crosswalk workflow in
``create_rsid_cbsa_cw.py`` but replaces the HUD/CBSA chain with direct
matches against the Geonames ``cities.txt`` reference. The resulting files
live in ``{root}/data/int``:

* ``rsid_geoname_cw.parquet`` — Revelio RSIDs to ``geoname_id``
* ``ipeds_geoname_cw.parquet`` — IPEDS UNITIDs to ``geoname_id``

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
        SELECT rsid, university_raw, COUNT(*) AS n
        FROM read_parquet('{WRDS_USERS_PATH}')
        WHERE university_raw IS NOT NULL
          AND rsid IS NOT NULL
          AND university_country = 'United States'
        GROUP BY rsid, university_raw
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


def map_rsid_to_geoname(univ_rsid: pd.DataFrame, school_lookup: pd.DataFrame) -> pd.DataFrame:
    univ_rsid = univ_rsid.copy()
    univ_rsid["univ_clean"] = _normalize(univ_rsid["university_raw"])
    univ_rsid = univ_rsid[univ_rsid["univ_clean"].ne("")]

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


def main() -> None:
    con = ddb.connect()

    geoname_variants = load_geoname_variants()
    geoname_sources = " and ".join(GEONAMES_CANDIDATE_FILES)
    print(f"Loaded {len(geoname_variants):,} geoname name variants for US cities from {geoname_sources}.")

    univ_zip_cw = load_univ_zip(con)
    univ_rsid_cw = load_univ_rsid(con)
    ipeds_cw = load_ipeds_crosswalk()

    print(f"FOIA schools: {len(univ_zip_cw):,} rows; RSID pairs: {len(univ_rsid_cw):,}; IPEDS names: {len(ipeds_cw):,}.")

    # matching FOIA school zips to geonames
    school_geo = match_city_state(univ_zip_cw, "campus_city", "campus_state", geoname_variants)
    resolved_school_share = school_geo["geoname_id"].notna().mean()
    print(f"Matched {resolved_school_share:.1%} of FOIA campus city/state pairs to geoname ids.")

    school_lookup = build_school_geoname_lookup(school_geo)
    rsid_geoname = map_rsid_to_geoname(univ_rsid_cw, school_lookup)
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

    con.close()


if __name__ == "__main__":
    main()
