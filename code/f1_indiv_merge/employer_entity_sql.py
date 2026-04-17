"""Shared SQL expressions for matching FOIA employer rows back to upstream entity ids."""

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


def sql_normalize_expr(col: str) -> str:
    return f"""
        TRIM(
          REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER({col}),
                      '[0-9]+/[0-9]+/[0-9]+', ' ', 'g'),
                    '[^a-z0-9 ]', ' ', 'g'
                ),
            '\\s+', ' ', 'g'
        ))
    """


def sql_clean_company_name_expr(col: str) -> str:
    suffix_regex = (
        "(?i)\\b("
        "inc|inc\\.|incorporated|llc|l\\.l\\.c|llp|l\\.l\\.p|lp|l\\.p|"
        "ltd|ltd\\.|limited|corp|corp\\.|corporation|company|co|co\\.|"
        "pllc|plc|pc|pc\\.|gmbh|ag|sa"
        ")\\b"
    )
    return sql_normalize_expr(f"REGEXP_REPLACE({col}, '{suffix_regex}', ' ', 'g')")


def sql_clean_zip_expr(col: str) -> str:
    zip_clean = f"TRIM(CAST(REGEXP_REPLACE({col}, '[^0-9]', '', 'g') AS VARCHAR))"
    return f"""
        CASE
            WHEN LENGTH(TRIM(CAST({zip_clean} AS VARCHAR))) = 4 THEN '0' || TRIM(CAST({zip_clean} AS VARCHAR))
            WHEN LENGTH(TRIM(CAST({zip_clean} AS VARCHAR))) >= 5 THEN SUBSTRING(TRIM(CAST({zip_clean} AS VARCHAR)) FROM 1 FOR 5)
            ELSE TRIM(CAST({zip_clean} AS VARCHAR))
        END
    """


def sql_state_name_to_abbr_expr(col: str) -> str:
    cases = " \n".join(
        [f"WHEN LOWER(TRIM({col})) = '{name}' THEN '{abbr}'" for name, abbr in STATE_NAME_TO_ABBR.items()]
    )
    return f"""
        CASE
            {cases}
            ELSE UPPER(TRIM({col}))
        END
    """
