# File Description: Matching Schools to OpenAlex to get Country
# Author: Amy Kim
# Date Created: Mon Jun 2 2025

# Imports and Paths
import argparse
import duckdb as ddb
import pandas as pd
import time
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from config import *
import rev_indiv_config as rcfg

con = ddb.connect()
run_tag = rcfg.RUN_TAG
t_script0 = time.time()
# Ensure logs stream to nohup output without long buffering delays.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
print(f"Using config: {rcfg.ACTIVE_CONFIG_PATH}", flush=True)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Match Revelio institutions to countries."
    )
    parser.add_argument(
        "--ipeds-only",
        action="store_true",
        help="Run only the IPEDS rematch step against an existing institution-country parquet.",
    )
    parser.add_argument(
        "--ipeds-input",
        default=None,
        help="Input parquet for --ipeds-only (default: configured final output path).",
    )
    parser.add_argument(
        "--ipeds-output",
        default=None,
        help="Output parquet for --ipeds-only (default: same as --ipeds-input).",
    )
    return parser.parse_args()


def _sql_escape_path(path: str) -> str:
    return str(path).replace("'", "''")


def _resolve_ipeds_crosswalk_path() -> str | None:
    candidates: list[str] = []
    try:
        from company_shift_share.config_loader import load_config as load_company_config

        company_cfg = load_company_config()
        company_paths = company_cfg.get("paths", {}) if isinstance(company_cfg, dict) else {}
        cfg_path = company_paths.get("ipeds_name_to_zip_crosswalk")
        if cfg_path:
            candidates.append(cfg_path)
    except Exception:
        pass

    candidates.extend(
        [
            f"{root}/data/int/ipeds_name_to_zip_crosswalk.parquet",
            f"{root}/data/int/ipeds_crosswalk.parquet",
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _build_ipeds_us_matches(
    univ_source_sql: str,
    *,
    output_table: str = "ipeds_us_matches",
    prefix: str = "rev_ipeds",
) -> None:
    ipeds_path = _resolve_ipeds_crosswalk_path()
    if not ipeds_path:
        log("IPEDS crosswalk not found; skipping IPEDS rematch step.")
        con.sql(
            f"""
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT
                NULL::BIGINT AS univ_id,
                NULL::VARCHAR AS university_raw,
                NULL::VARCHAR AS match_country,
                NULL::VARCHAR AS matchtype,
                NULL::BIGINT AS unitid
            WHERE FALSE
            """
        )
        return

    try:
        from company_shift_share import deps_foia_clean as foia_deps
    except Exception as exc:
        log(f"Could not import deps_foia_clean; skipping IPEDS rematch step ({exc}).")
        con.sql(
            f"""
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT
                NULL::BIGINT AS univ_id,
                NULL::VARCHAR AS university_raw,
                NULL::VARCHAR AS match_country,
                NULL::VARCHAR AS matchtype,
                NULL::BIGINT AS unitid
            WHERE FALSE
            """
        )
        return

    log(f"Running IPEDS rematch using: {ipeds_path}")
    con.sql(
        f"""
        CREATE OR REPLACE TABLE ipeds_crosswalk AS
        SELECT * FROM read_parquet('{_sql_escape_path(ipeds_path)}')
        """
    )
    foia_deps._create_ipeds_inst_view(con)
    con.sql(
        f"""
        CREATE OR REPLACE TABLE {prefix}_left AS
        SELECT
            univ_id,
            university_raw,
            {foia_deps._sql_clean_inst_name("university_raw")} AS rev_instname_clean
        FROM ({univ_source_sql})
        WHERE university_raw IS NOT NULL
        GROUP BY univ_id, university_raw
        """
    )
    match_views = foia_deps._build_inst_match_views(
        con,
        left_view=f"{prefix}_left",
        right_view="ipeds_inst",
        left_id_col="univ_id",
        right_id_col="UNITID",
        left_name_col="rev_instname_clean",
        right_name_col="ipeds_instname_clean",
        left_city_col=None,
        left_state_col=None,
        left_zip_col=None,
        right_city_col="ipeds_city_clean",
        right_state_col="ipeds_state_clean",
        right_zip_col="ipeds_zip_clean",
        right_alias_col="ipeds_alias",
        include_geo=False,
        include_city_in_name=False,
        include_subset=True,
        include_jw=True,
        token_fallback=True,
        token_top_n=150,
        token_min_len=3,
        rematch_jw_threshold=foia_deps.REMATCH_JW_THRESHOLD,
        prefix=prefix,
    )
    con.sql(
        f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH candidates AS (
            SELECT univ_id, right_id AS unitid, 'ipeds_direct' AS matchtype, 1 AS match_order
            FROM {match_views['good_matches']}
            UNION ALL
            SELECT univ_id, right_id AS unitid, 'ipeds_fuzzy' AS matchtype, 2 AS match_order
            FROM {match_views['good_second_matches']}
        ),
        ranked AS (SELECT univ_id, 'United States' AS match_country, matchtype, unitid
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER(PARTITION BY univ_id ORDER BY match_order, unitid) AS rn
            FROM candidates
        )
        WHERE rn = 1
        )
        SELECT ranked.univ_id, university_raw, ranked.match_country, ranked.matchtype, ranked.unitid FROM ranked JOIN {prefix}_left ON ranked.univ_id = {prefix}_left.univ_id
        """
    )
    n_ipeds = con.sql(f"SELECT COUNT(*) AS n FROM {output_table}").df().iloc[0, 0]
    log(f"Computed IPEDS US matches: {n_ipeds} rows")


def _run_ipeds_only(input_path: str, output_path: str) -> int:
    if not os.path.exists(input_path):
        log(f"Missing input file for --ipeds-only: {input_path}")
        return 1

    con.sql(
        f"""
        CREATE OR REPLACE TABLE final_matches_base AS
        SELECT * FROM read_parquet('{_sql_escape_path(input_path)}')
        """
    )
    base_cols = [str(c) for c in con.sql("DESCRIBE final_matches_base").df()["column_name"].tolist()]
    if "matchsource" not in base_cols:
        con.sql(
            """
            CREATE OR REPLACE TABLE final_matches_base AS
            SELECT *, 'pre_ipeds_only'::VARCHAR AS matchsource
            FROM final_matches_base
            """
        )
    _build_ipeds_us_matches(
        "SELECT ROW_NUMBER() OVER(ORDER BY university_raw) AS univ_id, university_raw FROM final_matches_base GROUP BY university_raw",
        output_table="ipeds_us_matches",
        prefix="rev_ipeds_only",
    )
    base_rows = int(con.sql("SELECT COUNT(*) AS n FROM final_matches_base").df().iloc[0, 0])
    base_univs = int(con.sql("SELECT COUNT(DISTINCT university_raw) AS n FROM final_matches_base").df().iloc[0, 0])
    base_exact_univs = int(con.sql("SELECT COUNT(DISTINCT university_raw) AS n FROM final_matches_base WHERE matchtype = 'exact'").df().iloc[0, 0])
    ipeds_univs = int(con.sql("SELECT COUNT(DISTINCT university_raw) AS n FROM ipeds_us_matches").df().iloc[0, 0])
    log(
        "IPEDS-only input summary: "
        f"rows={base_rows}, univs={base_univs}, exact_univs={base_exact_univs}, ipeds_matched_univs={ipeds_univs}"
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE final_matches AS
        WITH exact_univs AS (
            SELECT university_raw
            FROM final_matches_base
            WHERE matchtype = 'exact'
              AND university_raw IS NOT NULL
            GROUP BY university_raw
        ),
        ipeds_exact_raw AS (
            SELECT i.university_raw, i.match_country
            FROM ipeds_us_matches AS i
            LEFT JOIN exact_univs AS e
              ON i.university_raw = e.university_raw
            WHERE e.university_raw IS NULL
        ),
        ipeds_exact AS (
            SELECT
              university_raw,
              match_country,
              1 AS matchscore,
              'exact' AS matchtype,
              'ipeds_only_exact'::VARCHAR AS matchsource
            FROM ipeds_exact_raw
            GROUP BY university_raw, match_country
        ),
        ipeds_univs AS (
            SELECT university_raw FROM ipeds_exact GROUP BY university_raw
        )
        SELECT *
        FROM final_matches_base
        WHERE university_raw NOT IN (SELECT university_raw FROM ipeds_univs WHERE university_raw IS NOT NULL)
           OR matchtype = 'exact'
        UNION ALL
        SELECT university_raw, match_country, matchscore, matchtype, matchsource FROM ipeds_exact
        """
    )
    out_rows = int(con.sql("SELECT COUNT(*) AS n FROM final_matches").df().iloc[0, 0])
    out_exact_univs = int(con.sql("SELECT COUNT(DISTINCT university_raw) AS n FROM final_matches WHERE matchtype = 'exact'").df().iloc[0, 0])
    promoted_univs = int(
        con.sql(
            """
            SELECT COUNT(DISTINCT e.university_raw) AS n
            FROM (
                SELECT university_raw FROM ipeds_us_matches GROUP BY university_raw
            ) AS e
            LEFT JOIN (
                SELECT university_raw FROM final_matches_base WHERE matchtype = 'exact' GROUP BY university_raw
            ) AS b USING (university_raw)
            WHERE b.university_raw IS NULL
            """
        ).df().iloc[0, 0]
    )
    log(
        "IPEDS-only output summary: "
        f"rows={out_rows}, exact_univs={out_exact_univs}, newly_promoted_to_exact={promoted_univs}"
    )
    out_ipeds_exact = int(con.sql("SELECT COUNT(*) AS n FROM final_matches WHERE matchsource = 'ipeds_only_exact'").df().iloc[0, 0])
    log(f"IPEDS-only tagged rows (matchsource='ipeds_only_exact'): {out_ipeds_exact}")
    con.sql(f"COPY final_matches TO '{_sql_escape_path(output_path)}'")
    log(f"Saved IPEDS-only rematch output: {output_path}")
    return 0

openalex_match_filt_file = rcfg.OPENALEX_MATCH_FILT_PARQUET
dedup_match_filt_file = rcfg.DEDUP_MATCH_FILT_PARQUET
all_token_freqs_file = rcfg.ALL_TOKEN_FREQS_PARQUET
geonames_token_merge_file = rcfg.GEONAMES_TOKEN_MERGE_PARQUET
final_inst_country_file = rcfg.REV_INST_COUNTRIES_PARQUET

args = _parse_args()
if args.ipeds_only:
    ipeds_input = args.ipeds_input or final_inst_country_file
    ipeds_output = args.ipeds_output or ipeds_input
    rc = _run_ipeds_only(ipeds_input, ipeds_output)
    if rc != 0:
        raise SystemExit(rc)
    log(f"Done (IPEDS only). Total elapsed: {round((time.time() - t_script0)/60, 2)} min")
    raise SystemExit(0)

# Importing Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)
log(f"Loaded country crosswalk with {len(country_cw_dict)} keys")

## Creating DuckDB functions from python helpers
# country crosswalk function
con.create_function("get_std_country", lambda x: help.get_std_country(x, country_cw_dict), ['VARCHAR'], 'VARCHAR')

con.create_function("get_gmaps_country", lambda x: help.get_gmaps_country(x, country_cw_dict), ['VARCHAR'], 'VARCHAR')

####################
## IMPORTING DATA ##
####################
# Importing Data (From WRDS Server)
wrds_users_file = rcfg.WRDS_USERS_PARQUET
legacy_chunk_files = rcfg.LEGACY_WRDS_USER_MERGE_SHARDS
log("Starting data imports...")

if os.path.exists(wrds_users_file):
    log(f"Loading consolidated users file: {wrds_users_file}")
    rev_raw = con.read_parquet(wrds_users_file)
else:
    log("Consolidated users file not found. Falling back to legacy rev_user_merge shards.")
    rev_raw = con.read_parquet(legacy_chunk_files[0])
    for j in range(1, 10):
        rev_raw = con.sql(
            f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{legacy_chunk_files[j]}'"
        )
log(f"Loaded Revelio user data: {rev_raw.shape[0]} rows")

# Importing institutions data
institutions = con.read_csv(f"{root}/data/crosswalks/institutions.csv")
acronyms = con.read_csv(f"{root}/data/crosswalks/institutions_acronyms.csv")
altnames = con.read_csv(f"{root}/data/crosswalks/institutions_altnames.csv")

# Importing geonames data
cities500 = pd.read_csv(f"{root}/data/crosswalks/geonames/cities500.txt", 
                        sep = "\t",
                        names = ["geonameid", 'name','asciiname','altnernatenames','latitude','longitude','featureclass','featurecode','countrycode','cc2','admin1','admin2','admin3','admin4','pop','elev','dem','timezone','mod'],
                        keep_default_na = False)
admin1codes = pd.read_csv(f"{root}/data/crosswalks/geonames/admin1CodesASCII.txt",
                            sep = "\t",
                            names = ['code', 'name', 'asciiname', 'geonameid'])
admin2codes = pd.read_csv(f"{root}/data/crosswalks/geonames/admin2Codes.txt",
                            sep = "\t",
                            names = ['concatcodes', 'name', 'asciiname', 'geonameid'])

# Importing google maps data (matches from google maps)
gmaps = con.sql(f"""SELECT * FROM read_parquet('{"')UNION ALL SELECT * FROM read_parquet('".join([f"{root}/data/int/gmaps_univ_locations/{f}" for f in os.listdir(f"{root}/data/int/gmaps_univ_locations/")])}')""")

gmaps_clean = con.sql("SELECT top_country, top_country_n_users, university_raw, CASE WHEN gmaps_json IS NULL THEN NULL ELSE get_gmaps_country(gmaps_json.candidates[1].formatted_address) END AS univ_gmaps_country, gmaps_json.candidates[1].name AS gmaps_name, gmaps_json FROM gmaps")
log(f"Loaded gmaps matches: {gmaps_clean.shape[0]} rows")

###################
## CLEANING DATA ##
###################
# NOTE: FOR NOW, CLEANING STEP WILL REMOVE ALL NON-ASCII CHARACTERS AND PUNCTUATION (TODO: NEED TO USE TRANSLATE FOR UNMATCHED) 
#from deep_translator import GoogleTranslator
#GoogleTranslator(source = 'auto', target = 'en').translate('这是一支笔')

## CLEANING REVELIO
# Cleaning Degree and Institution Name
rev_clean = con.sql(
f"""
    SELECT 
    fullname, university_country, university_location, degree, user_id,
    {help.degree_clean_regex_sql()} AS degree_clean,
    {help.inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw 
"""
)
log(f"Built rev_clean: {rev_clean.shape[0]} rows")

# Collapsing to institution level (getting top non-null country)
#   first: filter out non-degree observations, then group by user x university x country (record indicator for whether high school) to get rid of users with multiple degrees from same inst
#   second: group by university x country (record number of users per cell, number of hs users per cell)
#   third: group by university, get top country (by number of users) and number of users of that top country, total number of users for institution, total share marked as HS
# MAIN THING: 
con.sql(
f"""CREATE OR REPLACE TABLE univ_names AS (SELECT university_raw, univ_raw_clean,
    (ARRAY_AGG(university_country ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country, 
    (ARRAY_AGG(n_users_univ_ctry ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country_n_users, 
    SUM(n_users_univ_ctry) AS n_users,    
    SUM(n_hs_ctry)/SUM(n_users_univ_ctry) AS share_hs,
    {help.inst_clean_withparan_regex_sql('university_raw')} AS univ_raw_clean_withparan,
    ROW_NUMBER() OVER() AS univ_id
FROM (
    SELECT university_raw, univ_raw_clean, university_country, COUNT(*) AS n_users_univ_ctry, SUM(hs) AS n_hs_ctry
    FROM (
        SELECT university_raw, univ_raw_clean, user_id, university_country, MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) AS hs 
        FROM rev_clean 
        WHERE degree_clean != 'Non-Degree'
        GROUP BY university_raw, univ_raw_clean, university_country, user_id
    ) GROUP BY university_country, university_raw, univ_raw_clean
) GROUP BY university_raw, univ_raw_clean ORDER BY n_users DESC)
""")
log("Created table univ_names")

# TEMPORARY THING FOR TESTING:
# con.sql(
# f"""CREATE OR REPLACE TABLE univ_names AS (SELECT university_raw, univ_raw_clean,
#     (ARRAY_AGG(university_country ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country, 
#     (ARRAY_AGG(n_users_univ_ctry ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country_n_users, 
#     SUM(n_users_univ_ctry) AS n_users,    
#     SUM(n_hs_ctry)/SUM(n_users_univ_ctry) AS share_hs,
#     {help.inst_clean_withparan_regex_sql('university_raw')} AS univ_raw_clean_withparan,
#     ROW_NUMBER() OVER() AS univ_id
# FROM (
#     SELECT university_raw, univ_raw_clean, university_country, COUNT(*) AS n_users_univ_ctry, SUM(hs) AS n_hs_ctry
#     FROM (
#         SELECT university_raw, univ_raw_clean, user_id, university_country, MAX(CASE WHEN degree_clean = 'High School' THEN 1 ELSE 0 END) AS hs 
#         FROM rev_clean 
#         WHERE degree_clean != 'Non-Degree'
#         GROUP BY university_raw, univ_raw_clean, university_country, user_id
#     ) GROUP BY university_country, university_raw, univ_raw_clean
# ) GROUP BY university_raw, univ_raw_clean ORDER BY RANDOM() LIMIT 100)
# """)

## CLEANING OPENALEX
# grouping by name, country, type, source (getting rid of duplicates), cleaning name
con.sql(
f"""
CREATE OR REPLACE TABLE inst_clean AS 
(SELECT *, CASE WHEN country_code = 'NA' THEN 'NA' ELSE get_std_country(country_code) END AS country_clean, ROW_NUMBER() OVER() AS inst_id, {help.inst_clean_regex_sql('name')} AS name_clean FROM (
    SELECT * FROM (
        SELECT id AS openalexid, name, country_code, type, 'institutions' AS source FROM institutions
        UNION ALL 
        SELECT id AS openalexid, alternative_names AS name, country_code, type, 'altnames' AS source FROM altnames 
        UNION ALL
        SELECT id AS openalexid, acronyms AS name, country_code, type, 'acronyms' AS source FROM acronyms
    )
    GROUP BY openalexid, name, country_code, type, source
))
""")
log("Created table inst_clean")

## CLEANING GEONAMES
cities500['pop_pctl'] = cities500['pop'].rank(pct=True)

# cleaning, getting alternate names and abbrevs, concatenating
geonames_agg = pd.concat([cities500.assign(type = 'city')[['name', 'countrycode', 'admin1', 'admin2', 'geonameid', 'type', 'pop', 'pop_pctl']],
           cities500.assign(name = cities500['altnernatenames'].str.split(','), originalname = cities500['name'])[['name','originalname','countrycode', 'admin1', 'admin2', 'geonameid', 'pop', 'pop_pctl']].explode('name').assign(type = 'city_altname'),
           admin1codes.assign(countrycode = admin1codes['code'].str[:2], admincode = admin1codes['code'].str[3:])[['name','countrycode', 'admincode', 'geonameid']].assign(type = 'admin1'),
           admin1codes.assign(countrycode = admin1codes['code'].str[:2], originalname = admin1codes['name'], name = admin1codes['code'].str[3:]).loc[(admin1codes['code'].str.match('^[A-Z]{2}\\..*[A-Z].*$')&(admin1codes['code'].str.len() > 4))][['name','originalname','countrycode', 'geonameid']].assign(type = 'admin1_abbrev'),
           admin2codes.assign(countrycode = admin2codes['concatcodes'].str[:2], admincode = admin2codes['concatcodes'].str.extract('[A-Z]{2}\\.([A-Z0-9]+\\.[A-z0-9]+)$'))[['name','countrycode', 'geonameid', 'admincode']].assign(type = 'admin2'),
           admin2codes.assign(countrycode = admin2codes['concatcodes'].str[:2], originalname = admin2codes['name'], name = admin2codes['concatcodes'].str.extract('[A-Z]{2}\\.[A-Z0-9]+\\.([A-z0-9]+)$')).loc[(admin2codes['concatcodes'].str.match('^[A-Z]{2}\\.[A-Z0-9]+\\..*[A-Z].*$') & admin2codes['concatcodes'].str.match('^[A-Z]{2}\\.[A-Z0-9]+\\.[A-Z0-9]{2,}$'))][['name', 'originalname','countrycode', 'geonameid']].assign(type = 'admin2_abbrev')
           ])

# standardizing country codes and merging with country names
countries = pd.DataFrame(country_cw_dict.items(), columns = ['name', 'countryname']).assign(type = 'country')

geonames_df = pd.concat([
    geonames_agg.assign(countryname = geonames_agg['countrycode'].map(country_cw_dict)),
    countries.assign(type = countries['type'].case_when([(countries['name'].str.match('[A-Z]{3}'), 'country_threeletter'),(countries['name'].str.match('[A-Z]{2}'), 'country_twoletter')]))[['name','countryname','type']]
])
log(f"Built geonames_df in pandas: {geonames_df.shape[0]} rows")

con.sql(f"CREATE OR REPLACE TABLE geonames AS SELECT *, {help.inst_clean_regex_sql('name')} AS name_clean FROM geonames_df")
log("Created table geonames")


#####################
## TOKENIZING DATA ##
#####################
# openalex institutions tokenized
openalex_tokens = con.sql(help.tokenize_sql('name_clean', 'inst_id', 'inst_clean'))

# revelio institutions tokenized with parenthetical text
rev_tokens_withparan = con.sql(help.tokenize_sql('univ_raw_clean_withparan', 'univ_id', 'univ_names', othercols = ',university_raw, univ_raw_clean_withparan'))

# revelio institutions tokenized without parenthetical text
rev_tokens_noparan = con.sql(help.tokenize_sql('univ_raw_clean', 'univ_id', 'univ_names'))
log(f"Tokenized names: openalex={openalex_tokens.shape[0]}, rev_with_paren={rev_tokens_withparan.shape[0]}, rev_no_paren={rev_tokens_noparan.shape[0]}")

# merging openalex and revelio (with paran) tokens to get agg freqs
all_tokens = con.sql("SELECT token, (COUNT(*) OVER(PARTITION BY token))/(COUNT(*) OVER()) AS token_freq, idnum, token_id, source, COUNT(token) OVER (PARTITION BY source, idnum) AS n_tokens FROM (SELECT token, token_id, idnum, 'revelio' AS source FROM rev_tokens_withparan UNION ALL SELECT token, token_id, idnum, 'openalex' AS source FROM openalex_tokens)")

# collapsing to token level
all_token1_freqs = con.sql("SELECT token, MEAN(token_freq) AS token_freq, ROW_NUMBER() OVER(ORDER BY MEAN(token_freq) DESC) AS token_rank, MAX(CASE WHEN source = 'openalex' THEN 1 ELSE 0 END) AS openalex, MAX(CASE WHEN source = 'revelio' THEN 1 ELSE 0 END) AS revelio FROM all_tokens GROUP BY token")
log(f"Computed token frequencies: {all_token1_freqs.shape[0]} unique tokens")

## USING TOKEN FREQUENCIES TO CORRECT SPELLING ERRORS
# getting 500 most common long words (over 3 letters)
spell_cw_jw = con.sql(
"""
    WITH rev_only AS 
        (SELECT * FROM all_token1_freqs WHERE openalex == 0 AND revelio == 1 AND LENGTH(token) > 3),
    keywords AS 
        (SELECT * FROM all_token1_freqs WHERE openalex == 1 AND revelio == 1 AND LENGTH(token) > 3 ORDER BY token_freq DESC)
    SELECT * FROM (
        SELECT a.token AS token, b.token AS token_corrected, a.token_freq AS token_freq, b.token_freq AS token_corrected_freq, b.token_rank AS token_corrected_rank, jaro_winkler_similarity(a.token, b.token) AS jwsim, ROW_NUMBER() OVER(PARTITION BY a.token ORDER BY jaro_winkler_similarity(a.token, b.token) DESC) AS matchord 
        FROM 
            (rev_only AS a JOIN keywords AS b ON jaro_winkler_similarity(a.token, b.token) >= 0.95))
    WHERE matchord = 1 AND token_corrected_rank < 500
"""
)

con.sql("CREATE OR REPLACE TABLE spell_cw_jw AS SELECT * FROM spell_cw_jw")
log(f"Created spell_cw_jw: {spell_cw_jw.shape[0]} rows")

# getting all words over 3 letters
spell_cw_jw_all = con.sql(
"""
    WITH rev_only AS 
        (SELECT * FROM all_token1_freqs WHERE openalex == 0 AND revelio == 1 AND LENGTH(token) > 3),
    keywords AS 
        (SELECT * FROM all_token1_freqs WHERE openalex == 1 AND revelio == 1 AND LENGTH(token) > 3 ORDER BY token_freq DESC)
    SELECT * FROM (
        SELECT a.token AS token, b.token AS token_corrected, a.token_freq AS token_freq, b.token_freq AS token_corrected_freq, b.token_rank AS token_corrected_rank, jaro_winkler_similarity(a.token, b.token) AS jwsim, ROW_NUMBER() OVER(PARTITION BY a.token ORDER BY jaro_winkler_similarity(a.token, b.token) DESC) AS matchord 
        FROM 
            (rev_only AS a JOIN keywords AS b ON jaro_winkler_similarity(a.token, b.token) >= 0.95))
"""
)

con.sql("CREATE OR REPLACE TABLE spell_cw_jw_all AS SELECT * FROM spell_cw_jw_all")
log(f"Created spell_cw_jw_all: {spell_cw_jw_all.shape[0]} rows")
# con.sql(f"CREATE OR REPLACE TABLE univ_names_spellcorr AS {help.spell_corr_sql('spell_cw_jw','univ_names','univ_raw_clean','univ_id')}")

################################
## EXTRACTING 'EXACT' MATCHES ##
################################
# match on cleaned & spell-corrected names
# cleanmatches = con.sql(
# """
#     SELECT * FROM 
#         ((SELECT * FROM univ_names_spellcorr WHERE LENGTH(univ_raw_clean) > 2) AS a 
#         JOIN (SELECT * FROM inst_clean WHERE LENGTH(name_clean) > 2 AND type != 'company') AS b 
#         ON a.univ_raw_clean_corr = b.name_clean)""")

# extracting parenthetical text
univ_names_parenth = con.sql("SELECT *, regexp_extract_all(university_raw, '(\\(|\\[)([^\\)\\]]*)(\\)|\\])', 2) AS parenth_list, regexp_replace(university_raw, '\\s*(\\(|\\[)[^\\)\\]]*(\\)|\\])\\s*', ' ', 'g') AS university_raw_nopar FROM univ_names")

# combining raw text (no parenth) + parenth text
univ_names_parenth_split = con.sql("SELECT university_raw_nopar AS univ_text, university_raw, univ_id FROM univ_names_parenth WHERE university_raw_nopar !~ '\\s' AND university_raw_nopar != '' UNION ALL SELECT unnest(parenth_list) AS univ_text, university_raw, univ_id FROM univ_names_parenth WHERE len(parenth_list) > 0")

# breaking up raw text by comma (only if comma exists) from both raw and parenth text
comma_tokens = con.sql(
    f"""
    SELECT token AS comma_token, idnum AS univ_id, university_raw FROM ({help.tokenize_sql('univ_text', 'univ_id', "(SELECT * FROM univ_names_parenth_split WHERE univ_text ~ '.+,.+')", sep = "','", othercols = ", university_raw")}) WHERE token !~ '\\s+'
    """)
    
# combining all types of clean text (raw, parenth, comma tokens) 
univ_names_combined = con.sql(f"SELECT univ_id, university_raw, univ_text, CASE WHEN univ_text_clean = '' AND univ_text != '' THEN univ_text ELSE univ_text_clean END AS univ_text_clean, ROW_NUMBER() OVER() AS univ_combined_id FROM (SELECT *, {help.inst_clean_regex_sql('univ_text')} AS univ_text_clean FROM (SELECT * FROM univ_names_parenth_split UNION ALL SELECT comma_token AS univ_text, university_raw, univ_id FROM comma_tokens))")

con.sql("CREATE OR REPLACE TABLE univ_names_combined AS (SELECT * FROM univ_names_combined)")
log(f"Created univ_names_combined: {univ_names_combined.shape[0]} rows")

# and spell-correcting (columns: univ_id, university_raw, univ_text (raw text to match on, could be from comma tokens or parenth or full university raw), univ_text_clean (cleaned text version of univ_text), univ_combined_id (unique id) = idnum, univ_text_clean_corr (cleaned and spell corrected version of univ_text))
con.sql(f"CREATE OR REPLACE TABLE univ_names_spellcorr AS {help.spell_corr_sql('spell_cw_jw','univ_names_combined','univ_text_clean','univ_combined_id')}")
log("Created table univ_names_spellcorr")

# matching
allmatches = con.sql(
"""
    SELECT *, COUNT(DISTINCT openalexid) OVER(PARTITION BY c.univ_id) AS nmatch, COUNT(DISTINCT country_clean) OVER(PARTITION BY c.univ_id) AS ncountry, COUNT(DISTINCT openalexid) OVER(PARTITION BY univ_text_clean_corr) AS nmatch_jointext FROM 
        ((SELECT univ_text_clean_corr, univ_id FROM univ_names_spellcorr WHERE LENGTH(univ_text_clean) > 2) AS a 
        JOIN (SELECT openalexid, name_clean, FIRST(name) AS name, FIRST(country_clean) AS country_clean FROM (SELECT * FROM inst_clean WHERE LENGTH(name_clean) > 2 AND type != 'company' AND source != 'acronyms') GROUP BY openalexid, name_clean) AS b 
        ON a.univ_text_clean_corr = b.name_clean) AS c LEFT JOIN univ_names AS d ON c.univ_id = d.univ_id""")

# matches to keep
exactmatches = con.sql("SELECT univ_id, openalexid AS matchid, university_raw, name AS matchname_raw, univ_text_clean_corr AS rev_clean, name_clean AS matchname_clean, country_clean AS match_country, 1 AS matchscore, 'exact_openalex' AS matchtype FROM allmatches WHERE ncountry = 1")
log(f"Computed exactmatches: {exactmatches.shape[0]} rows")


##########################################
## MATCHING TO CITY LOCATIONS ON TOKENS ##
##########################################
# list of tokens to remove from citynames
city_tokens = ['ba', 'to', 'in', 'of', 'and', 'university', 'academy', 'college', 'mba', 'bsc', 'the', 'st', 'area', 'center', 'universidad', 'central', 'central high', 'at', 'valley', 'jr', 'sr']
abbrev_tokens = ['DE', 'IN', 'BE', 'AT', 'ST', 'EN']

ids = ",".join(
    con.sql(
        help.random_ids_sql(
            "univ_id", "univ_names", rcfg.CLEAN_INST_RANDOM_UNIV_SAMPLE_N
        )
    )
    .df()["univ_id"]
    .astype(str)
)
test = rcfg.CLEAN_INST_TEST

# cleaning geographies: joining geonames data on itself to get full names of admin1 and admin2 + three-letter country codes for cities
geo_clean = con.sql(
f"""SELECT 
        name, a.countrycode, admin1, admin2, geonameid, type, originalname, a.countryname, country3code, name_clean, admin1name, admin1name_clean, admin2name, admin2name_clean, pop, pop_pctl, LENGTH(REGEXP_REPLACE(name_clean, '\\s+', ' ', 'g')) - LENGTH(REGEXP_REPLACE(name_clean, '\\s+', '', 'g')) AS n_spaces 
        FROM (
            geonames AS a 
            LEFT JOIN 
            (SELECT name AS admin1name, name_clean AS admin1name_clean, countrycode, admincode FROM geonames WHERE type = 'admin1') AS b 
            ON a.countrycode = b.countrycode AND a.admin1 = b.admincode 
            LEFT JOIN 
            (SELECT 
                name as admin2name, name_clean AS admin2name_clean, countrycode, REGEXP_EXTRACT(admincode, '([A-z0-9]+)\\.([A-z0-9]+)$', 1) AS admin1code, REGEXP_EXTRACT(admincode, '([A-z0-9]+)\\.([A-z0-9]+)$', 2) AS admin2code 
            FROM geonames WHERE type = 'admin2') AS c 
            ON a.countrycode = c.countrycode AND a.admin1 = c.admin1code AND a.admin2 = c.admin2code
            LEFT JOIN 
            (SELECT name AS country3code, countryname FROM geonames WHERE type = 'country_threeletter' AND country3code NOT IN ('COD', 'COG', 'ARE', 'XKX', 'PRI')) AS d 
            ON a.countryname = d.countryname
        ) WHERE name_clean NOT IN ('{"','".join(city_tokens)}') 
            AND LENGTH(name_clean) > 1 
            AND NOT (type = 'city_altname' AND name_clean = ({help.inst_clean_regex_sql('originalname')})) """)
log(f"Prepared geo_clean: {geo_clean.shape[0]} rows")

# iterating through to match by number of words in city/country name
for i in range(8):
    if test:
        tab = f"(SELECT * FROM univ_names WHERE univ_id IN ({ids}))"
    else:
        tab = "univ_names"
    rev_tokens = con.sql(help.tokenize_nword_sql(col = 'univ_raw_clean_withparan',
                                                 id = 'univ_id',
                                                 tab = tab,
                                                 n = i + 1,
                                                 othercols = ', university_raw, univ_raw_clean_withparan'))
    
    join_geonames = con.sql(
        f"""
        SELECT token, token_id, idnum AS univ_id, university_raw, univ_raw_clean_withparan,   
            name, countrycode, country3code, countryname, admin1, admin1name, admin1name_clean, admin2, admin2name, admin2name_clean, geonameid, type, originalname, pop, pop_pctl, n_spaces 
        FROM (
            SELECT token, MAX(token_id) AS token_id, idnum, university_raw, univ_raw_clean_withparan FROM rev_tokens GROUP BY token, idnum, university_raw, univ_raw_clean_withparan
        ) AS a JOIN (
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY geonameid, name_clean ORDER BY name) AS rn FROM geo_clean
            ) WHERE rn = 1 AND n_spaces = {i}
        ) AS b ON a.token = b.name_clean
        """)

    # at this stage, matches should be unique on the univ_id x geonameid x token level
    print(f"Total number of matches with {i} spaces: {join_geonames.shape[0]}", flush=True)

    if i == 0:
        con.sql(f"CREATE OR REPLACE TABLE geoname_matches AS SELECT *, {i} AS token_n FROM join_geonames") 
    else:
        con.sql(f"CREATE OR REPLACE TABLE geoname_matches AS SELECT * FROM geoname_matches UNION ALL SELECT *, {i} AS token_n FROM join_geonames")
log("Finished iterative geoname matching")

# goal: identify 
# filtering matches on whether another level of geography matches (country, admin1, admin2)

# pivot matches long on other geographies (each row is a tokenmatch x geography type, e.g. for tokenmatch with paris, pivot long to paris x FR, paris X France, paris x Ile de France, etc.)
geonames_long = con.sql("SELECT token, univ_id, university_raw, country, univ_raw_clean_withparan, geonameid, type, geoname, geotype, pop, pop_pctl, token_n, LENGTH(REGEXP_REPLACE(univ_raw_clean_withparan, '\\s+', ' ', 'g')) - LENGTH(REGEXP_REPLACE(univ_raw_clean_withparan, '\\s+', '', 'g')) + 1 - token_n - token_id AS tokens_from_end FROM (UNPIVOT (SELECT *, countryname AS country FROM geoname_matches) ON countrycode, country3code, countryname, admin1, admin1name_clean, admin2, admin2name_clean INTO NAME geotype VALUE geoname) WHERE geoname != '' AND geoname IS NOT NULL")

# scoring matches: 
#       - problem_abbrev_ind flags when abbreviations (in, de, be) overlap with common tokens, requires match to be capitalized otherwise flagged and matchtype_score replaced with 0.1
#       - matchtype_score -- 1 if inst name contains 'city othergeo' in that order, 0.5 if inst name contains othergeo but in different order, otherwise 0.1
#       - geotype_score modifies matchtype_score by factor of 1 if othergeo is full name of country, admin1 or admin2, 0.9 if alpha character admin1/2 abbrev, 0.8 otherwise
#       - score computed as location_score * length_score * geoscore
geonames_citymatch = con.sql(
f"""
SELECT *,
    CASE WHEN problem_abbrev_ind = 0 THEN 0.1*geotype2_score 
        ELSE matchtype_score*geotype2_score 
        END AS matchtype_score_corr,
    location_score * length_score * geoscore AS score
FROM 
    (SELECT *, 
    -- augmenting match score with indicator for if second geo is 'problematic' (then requiring a match on capitalized geo)
        CASE WHEN geoname NOT IN ('{"','".join(abbrev_tokens)}') THEN 1
            WHEN (university_raw LIKE '%' || geoname || '%') THEN 1
                ELSE 0
                END AS problem_abbrev_ind,
    -- match type score: ordered match ('...token, geo2') better than unordered match ('...geo2...token') better than no additional match 
        CASE 
            WHEN
                ((univ_raw_clean_withparan LIKE '% ' || token || ' ' || lower(geoname) || ' %') OR 
                (univ_raw_clean_withparan LIKE '% ' || token || ' ' || lower(geoname)) OR 
                (univ_raw_clean_withparan LIKE token || ' ' || lower(geoname) || ' %')) AND
                (lower(geoname) != token)
                THEN 1
            WHEN 
                ((univ_raw_clean_withparan LIKE '% ' || lower(geoname) || ' %') OR
                (univ_raw_clean_withparan LIKE '% ' || lower(geoname) ) OR
                (univ_raw_clean_withparan LIKE lower(geoname) || ' %')) AND 
                NOT ((token LIKE '%' || lower(geoname) || '%') OR (lower(geoname) LIKE '%' || token || '%'))
                THEN 0.5
            ELSE 0.1 
            END AS matchtype_score,
    -- geotype2: if secondary geo match, then ranking match quality based on geotype
        CASE 
            WHEN geotype IN ('countryname', 'admin1name_clean', 'admin2name_clean') THEN 1
            WHEN (geotype = 'admin1' AND geoname ~ '.*[A-z].*' ) OR 
                (geotype = 'admin2' AND geoname ~ '.*[A-z].*') THEN 0.9
            ELSE 0.8
            END AS geotype2_score,
    -- location score: likely match if last token (and not city), also likely match if after comma
        CASE 
            WHEN NOT (type = 'city' OR type = 'city_altname') AND tokens_from_end = 0 THEN 1
            WHEN lower(university_raw) LIKE '%,%' || token || '%' THEN 0.7
            ELSE 0.5
            END AS location_score,
    -- length score: better match if longer token, if token very short (<3 letters) check if capitalized
        CASE
            WHEN LENGTH(token) <= 3 AND NOT (university_raw LIKE '%' || upper(token) || '%') THEN (CASE WHEN LENGTH(token) <= 2 THEN 0.1 ELSE 0.2 END)
            WHEN token_n = MAX(token_n) OVER(PARTITION BY univ_id) THEN 1
            WHEN LENGTH(token) >= (MAX(LENGTH(token)) OVER(PARTITION BY univ_id) - 3) THEN 0.9
            ELSE 0.8 
            END AS length_score,
    -- geo score: priority to city, country, us state, then city altname, other admin, then everything else (all abbrevs)
        CASE
            WHEN type = 'city' THEN 1
            WHEN type ='country' THEN 1
            WHEN type = 'admin1' AND country = 'United States' THEN 1
            WHEN type = 'admin1' THEN 0.9
            WHEN type = 'admin2' THEN 0.9
            WHEN type = 'city_altname' THEN 0.9
            WHEN type = 'admin1_abbrev' THEN 0.8
            WHEN type = 'admin2_abbrev' THEN 0.8
            WHEN type = 'country_twoletter' THEN 0.8
            ELSE 0.8
        END AS geoscore
    FROM geonames_long)
""")

# pivoting back to univ_id x geonameid x token level and keeping max score across all geotypes
con.sql("""
CREATE OR REPLACE TABLE geomatch_wide AS
    (SELECT univ_id, university_raw, token, match_score_max AS match_score, score, token_n, geoscore, tokens_from_end, countryname, MAX(match_score_max) OVER(PARTITION BY univ_id) AS match_score_max, LENGTH(token) AS token_length FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY geonameid, univ_id ORDER BY match_score_max DESC) AS tokenrank FROM (
            PIVOT (
                SELECT university_raw, univ_id, geonameid, token, geoname, geotype, geoscore, tokens_from_end, type, token_n, score, length_score, location_score,
                    MAX(matchtype_score_corr) OVER(PARTITION BY geonameid, univ_id, token) AS match_score_max
                FROM geonames_citymatch
            ) ON geotype USING first(geoname)
        )
    )
WHERE tokenrank = 1)""")

all_geomatches = con.sql(
"""
SELECT *, COUNT(DISTINCT countryname) OVER(PARTITION BY univ_id) AS ncountry FROM (
    SELECT *
    FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY univ_id, countryname ORDER BY score DESC, tokens_from_end, token_length) AS match_rank
        FROM geomatch_wide WHERE match_score_max = match_score
    ) WHERE match_rank = 1 AND score >= 0.1
)
""")

exact_geomatches = con.sql("SELECT * FROM all_geomatches WHERE match_score >= 0.5 AND ncountry = 1")
log(f"Computed geomatches: all={all_geomatches.shape[0]}, exact={exact_geomatches.shape[0]}")

# IPEDS institution rematch (US assignment only; follows deps_foia_clean matching utilities)
_build_ipeds_us_matches(
    "SELECT univ_id, university_raw FROM univ_names",
    output_table="ipeds_us_matches",
    prefix="rev_ipeds",
)

################################
## MATCHING TO OPENALEX ON TOKENS ##
################################
univ_names_formatch = con.sql(
"""
SELECT
    a.univ_id,
    university_raw,
    univ_raw_clean_withparan,
    CASE
        WHEN b.exactmatch IS NOT NULL THEN 'exactmatch'
        WHEN d.ipedsmatch IS NOT NULL THEN 'ipedsmatch'
        WHEN c.citymatch IS NOT NULL THEN 'citymatch'
        ELSE 'nomatch'
    END AS matchind,
    CASE
        WHEN b.exactmatch IS NOT NULL THEN 'exactmatch'
        WHEN d.ipedsmatch IS NOT NULL THEN 'ipedsmatch'
        WHEN c.citymatch IS NOT NULL THEN 'citymatch'
        WHEN top_country_n_users >= 20 THEN 'nativematch'
        ELSE 'nomatch'
    END AS dedupind
FROM univ_names AS a
LEFT JOIN (SELECT univ_id, 1 AS exactmatch FROM exactmatches GROUP BY univ_id) AS b
    ON a.univ_id = b.univ_id
LEFT JOIN (SELECT univ_id, 1 AS citymatch FROM exact_geomatches GROUP BY univ_id) AS c
    ON a.univ_id = c.univ_id
LEFT JOIN (SELECT univ_id, 1 AS ipedsmatch FROM ipeds_us_matches GROUP BY univ_id) AS d
    ON a.univ_id = d.univ_id
"""
)
log(f"Prepared univ_names_formatch: {univ_names_formatch.shape[0]} rows")

tokenmatch = rcfg.CLEAN_INST_TOKENMATCH
if tokenmatch: 
    log("Starting token-level OpenAlex/dedup matching (tokenmatch=True)")
    con.sql("SELECT dedupind, COUNT(*) FROM univ_names_formatch GROUP BY dedupind")

    # con.sql(f"CREATE OR REPLACE TABLE revsamp AS (SELECT * FROM univ_names ORDER BY RANDOM() LIMIT 5000)")

    con.sql(f"CREATE OR REPLACE TABLE univ_names_matchind AS SELECT * FROM univ_names_formatch")
            
    # code structure: start with i = n, tokenize and match all univ_ids on i-word tokens
    #   increment i = i - 1 and tokenize and match univ_ids not previously matched on i-word tokens (i-1 word matches will be weakly worse than i word matches)
    firstflag = 1
    for i in range(8,1,-1):
        print(f"i = {i}", flush=True)
        t1 = time.time()
        rev_tokens = con.sql(help.tokenize_nword_sql(col = 'univ_raw_clean_withparan', 
                                                    id = 'univ_id',
                                                    tab = 'univ_names',
                                                    n = i,
                                                    othercols = ', univ_raw_clean_withparan, university_raw, top_country'))
        
        # getting total number of unique univids 
        rev_n_i = con.sql(f"SELECT * FROM univ_names WHERE LENGTH(REGEXP_REPLACE(univ_raw_clean_withparan, '\\s+', ' ', 'g')) - LENGTH(REGEXP_REPLACE(univ_raw_clean_withparan, '\\s+', '', 'g')) >= {i-1}").shape[0]

        openalex_tokens = con.sql(help.tokenize_nword_sql(col = 'name_clean',
                                                        id = 'inst_id',
                                                        tab = 'inst_clean',
                                                        n = i,
                                                        othercols = ', name_clean, name, openalexid, country_clean'))

        oa_n_i = con.sql(f"SELECT * FROM inst_clean WHERE LENGTH(REGEXP_REPLACE(name_clean, '\\s+', ' ', 'g')) - LENGTH(REGEXP_REPLACE(name_clean, '\\s+', '', 'g')) >= {i-1}").shape[0]

        # token frequencies 
        token_freqs = con.sql(f"SELECT token, COUNT(*)/MEAN(n) AS token_freq, COUNT(*) AS token_count, ROW_NUMBER() OVER(ORDER BY token_freq DESC) AS token_rank FROM (SELECT token, COUNT(*) OVER() AS n FROM (SELECT token FROM rev_tokens UNION ALL SELECT token FROM openalex_tokens)) GROUP BY token ORDER by token_freq DESC")

        if i == 2:
            c = 0.0005
        else:
            c = 0.004

        # matching revelio and openalex tokens with token freqs to get rid of common tokens
        #   also matching revelio tokens with univ_names_matchind
        rev_tokens_withfreqs = con.sql(
            f"""
            SELECT idnum AS univ_id, token, token_freq, token_count, a.univ_raw_clean_withparan, a.university_raw, top_country, matchind, dedupind FROM (
                SELECT * FROM (
                    SELECT a.token, token_freq, idnum, univ_raw_clean_withparan, university_raw, top_country, token_count 
                    FROM rev_tokens AS a 
                    LEFT JOIN token_freqs AS b 
                    ON a.token = b.token
                ) WHERE token_count < {(rev_n_i + oa_n_i)*c}
            ) AS a
            LEFT JOIN univ_names_matchind AS b
            ON a.idnum = b.univ_id
            """)
        con.sql(f"CREATE OR REPLACE TABLE rev_tokens_withfreqs_{i} AS SELECT * FROM rev_tokens_withfreqs")

        openalex_tokens_withfreqs = con.sql(f"SELECT * FROM (SELECT a.token, token_freq, idnum, name, name_clean, country_clean, openalexid, token_count FROM openalex_tokens AS a LEFT JOIN token_freqs AS b ON a.token = b.token) WHERE token_count < {(rev_n_i + oa_n_i)*c}")
        con.sql(f"CREATE OR REPLACE TABLE openalex_tokens_withfreqs_{i} AS SELECT * FROM openalex_tokens_withfreqs")

        # matchin tokens from previously unmatched institutions with openalex
        con.sql(
        f"""CREATE OR REPLACE TABLE token_match_{i} AS (SELECT univ_id, a.token AS rev_token, c.token AS openalex_token, 
                jaro_similarity(a.token, c.token) AS tokensim, 
                univ_raw_clean_withparan, university_raw, 
                idnum AS inst_id, name_clean, name, openalexid, top_country, country_clean
            FROM (SELECT * FROM rev_tokens_withfreqs_{i} WHERE matchind = 'nomatch') AS a 
                JOIN openalex_tokens_withfreqs_{i} AS c 
                ON jaro_similarity(a.token, c.token) >= 0.95)"""
        )

        # matching tokens back to revelio (deduplication)
        con.sql(
        f"""CREATE OR REPLACE TABLE dedup_match_{i} AS (SELECT a.univ_id AS left_univ_id, c.univ_id AS right_univ_id, a.token AS left_token, c.token AS right_token, 
                jaro_similarity(a.token, c.token) AS tokensim, 
                a.univ_raw_clean_withparan AS left_univ_raw_clean_withparan,
                c.univ_raw_clean_withparan AS right_univ_raw_clean_withparan,
                a.university_raw AS left_university_raw,
                c.university_raw AS right_university_raw 
            FROM (SELECT * FROM rev_tokens_withfreqs_{i} WHERE dedupind = 'nomatch') AS a 
                JOIN (SELECT * FROM rev_tokens_withfreqs_{i} WHERE dedupind IN ('exactmatch', 'ipedsmatch', 'citymatch', 'nativematch')) AS c 
                ON jaro_similarity(a.token, c.token) >= 0.95 AND NOT a.univ_id = c.univ_id)"""
        )

        # updating token_match and token_freq database
        if firstflag:
            con.sql(f"CREATE OR REPLACE TABLE all_token_matches AS SELECT *, {i} AS token_n FROM token_match_{i}") 
            print('token matches saved', flush=True)

            con.sql(f"CREATE OR REPLACE TABLE all_dedup_matches AS SELECT *, {i} AS token_n FROM dedup_match_{i}") 
            print('dedup matches saved', flush=True)

            con.sql(f"CREATE OR REPLACE TABLE all_token_freqs AS SELECT *, {i} AS token_n FROM token_freqs") 
            print('token freqs saved', flush=True)

            firstflag = 0

        else:
            con.sql(f"CREATE OR REPLACE TABLE all_token_matches AS SELECT * FROM all_token_matches UNION ALL SELECT *, {i} AS token_n FROM token_match_{i}")
            # print(con.sql("SELECT COUNT(*), token_n FROM all_token_matches GROUP BY token_n"))
            print('token matches saved', flush=True)
            
            con.sql(f"CREATE OR REPLACE TABLE all_dedup_matches AS SELECT * FROM all_dedup_matches UNION ALL SELECT *, {i} AS token_n FROM dedup_match_{i}")
            print('dedup matches saved', flush=True)

            con.sql(f"CREATE OR REPLACE TABLE all_token_freqs AS SELECT * FROM all_token_freqs UNION ALL SELECT *, {i} AS token_n FROM token_freqs")
            print('token freqs saved', flush=True)


        # updating univ_names_matchind database
        #   this version marks as matched only if matched with jaro similarity of 1
        #  con.sql(f"CREATE OR REPLACE TABLE univ_names_matchind AS (SELECT a.univ_id, university_raw, univ_raw_clean_withparan, CASE WHEN b.tokenmatch IS NOT NULL THEN 'token{i}match' ELSE matchind END AS matchind FROM univ_names_matchind AS a LEFT JOIN (SELECT univ_id, 1 AS tokenmatch FROM token_match WHERE tokensim = 1 GROUP BY univ_id) AS b ON a.univ_id = b.univ_id)")
        #   this version marks as matched no matter what
        con.sql(f"CREATE OR REPLACE TABLE univ_names_matchind AS (SELECT a.univ_id, university_raw, univ_raw_clean_withparan, CASE WHEN b.tokenmatch IS NOT NULL THEN 'token{i}match' ELSE matchind END AS matchind, CASE WHEN c.dedupmatch IS NOT NULL THEN 'token{i}match' ELSE dedupind END AS dedupind FROM univ_names_matchind AS a LEFT JOIN (SELECT univ_id, 1 AS tokenmatch FROM token_match_{i} GROUP BY univ_id) AS b ON a.univ_id = b.univ_id LEFT JOIN (SELECT left_univ_id, 1 AS dedupmatch FROM dedup_match_{i} GROUP BY left_univ_id) AS c ON a.univ_id = c.left_univ_id)")

        print(con.sql("SELECT COUNT(*), matchind FROM univ_names_matchind GROUP BY matchind"), flush=True)

        t2 = time.time()
        print(f"Time for iter {i}: {t2-t1}s", flush=True)

    # cleaning results (grouped to get rid of duplicate tokens, then grouped again to get rev x openalex pairs with multiple token matches, flagging and taking rarer token freq)
    freq_buffer = 0.1
    sim_buffer = 0.05

    match_res = con.sql(
    f"""
    SELECT *, 
        CASE WHEN log_token_freq <= (MIN(log_token_freq) OVER(PARTITION BY univ_id)) + {freq_buffer} THEN 1 ELSE 0 END AS min_freq,
        CASE WHEN jaro_sim >= (MAX(jaro_sim) OVER(PARTITION BY univ_id)) - {sim_buffer} THEN 1 ELSE 0 END AS max_jaro_sim,
        CASE WHEN  dl_sim >= (MAX(dl_sim) OVER(PARTITION BY univ_id)) - {sim_buffer} THEN 1 ELSE 0 END AS max_dl_sim
    FROM (
        SELECT univ_id, openalexid, 
            university_raw, name, univ_raw_clean_withparan, name_clean, token_n, top_country, country_clean,
            jaro_similarity(name_clean, univ_raw_clean_withparan) AS jaro_sim,
            1 - damerau_levenshtein(name_clean, univ_raw_clean_withparan)/(CASE WHEN LENGTH(name_clean) > LENGTH(univ_raw_clean_withparan) THEN LENGTH(name_clean) ELSE LENGTH(univ_raw_clean_withparan) END) AS dl_sim,
            COUNT(*) AS n_token_matches,
            ARRAY_AGG(CASE WHEN rev_token_freq > openalex_token_freq THEN LOG(rev_token_freq) ELSE LOG(openalex_token_freq) END ORDER BY CASE WHEN rev_token_freq > openalex_token_freq THEN LOG(rev_token_freq) ELSE LOG(openalex_token_freq) END)[1] AS log_token_freq,
            ARRAY_AGG(a.rev_token ORDER BY CASE WHEN rev_token_freq > openalex_token_freq THEN LOG(rev_token_freq) ELSE LOG(openalex_token_freq) END)[1] AS rev_token,
            ARRAY_AGG(a.openalex_token ORDER BY CASE WHEN rev_token_freq > openalex_token_freq THEN LOG(rev_token_freq) ELSE LOG(openalex_token_freq) END)[1] AS openalex_token,
            CASE WHEN (name_clean LIKE '%' || univ_raw_clean_withparan || '%' OR univ_raw_clean_withparan LIKE '%' || name_clean || '%') THEN 1 ELSE 0 END AS namecontain
        FROM (
            (SELECT univ_id, openalexid, inst_id, rev_token, openalex_token, tokensim, university_raw, univ_raw_clean_withparan, name, name_clean, token_n, top_country, country_clean FROM all_token_matches GROUP BY univ_id, openalexid, inst_id, rev_token, openalex_token, tokensim, university_raw, univ_raw_clean_withparan, name, name_clean, token_n, top_country, country_clean) AS a 
            LEFT JOIN (
                SELECT token AS rev_token, token_rank AS rev_token_rank, token_freq AS rev_token_freq FROM all_token_freqs
            ) AS b ON a.rev_token = b.rev_token
            LEFT JOIN (
                SELECT token AS openalex_token, token_rank AS openalex_token_rank, token_freq AS openalex_token_freq FROM all_token_freqs
            ) AS c ON a.openalex_token = c.openalex_token
        )
        GROUP BY univ_id, openalexid, inst_id, university_raw, univ_raw_clean_withparan, name, name_clean, token_n, top_country, country_clean
    )
    """)

    # next step: filtering out based on min_freq, max_jaro_sim, max_dl_sim, namecontain
    match_filt = con.sql("SELECT * FROM match_res WHERE (min_freq AND (max_jaro_sim OR max_dl_sim)) OR namecontain")


    # same for dedup matches
    dedup_res = con.sql(
    f"""
    SELECT *, 
        CASE WHEN log_token_freq <= (MIN(log_token_freq) OVER(PARTITION BY left_univ_id)) + {freq_buffer} THEN 1 ELSE 0 END AS min_freq,
        CASE WHEN jaro_sim >= (MAX(jaro_sim) OVER(PARTITION BY left_univ_id)) - {sim_buffer} THEN 1 ELSE 0 END AS max_jaro_sim,
        CASE WHEN  dl_sim >= (MAX(dl_sim) OVER(PARTITION BY left_univ_id)) - {sim_buffer} THEN 1 ELSE 0 END AS max_dl_sim
    FROM (
        -- creating variables for number of token matches within a univ-univ pair and row number within that pair (taking token pair with largest number of words first, then as tie breaker taking the rarest token pair)
        -- also creating a bunch of match measures
        SELECT *, 
            COUNT(*) OVER(PARTITION BY left_university_raw, right_university_raw) AS n_token_matches,
            ROW_NUMBER() OVER(PARTITION BY left_university_raw, right_university_raw ORDER BY token_n DESC, log_token_freq) AS token_rank,
            jaro_similarity(left_univ_raw_clean_withparan, right_univ_raw_clean_withparan) AS jaro_sim,
            1 - damerau_levenshtein(left_univ_raw_clean_withparan, right_univ_raw_clean_withparan)/(CASE WHEN LENGTH(left_univ_raw_clean_withparan) > LENGTH(right_univ_raw_clean_withparan) THEN LENGTH(left_univ_raw_clean_withparan) ELSE LENGTH(right_univ_raw_clean_withparan) END) AS dl_sim,
            CASE WHEN (left_univ_raw_clean_withparan LIKE '%' || right_univ_raw_clean_withparan || '%' OR right_univ_raw_clean_withparan LIKE '%' || left_univ_raw_clean_withparan || '%') THEN 1 ELSE 0 END AS namecontain
        FROM (
        -- creating variable equal to most common token between left and right
            SELECT *, CASE WHEN left_token_freq > right_token_freq THEN LOG(left_token_freq) ELSE LOG(right_token_freq) END AS log_token_freq FROM (
                (SELECT DISTINCT left_univ_id, right_univ_id, left_token, right_token, tokensim, left_university_raw, left_univ_raw_clean_withparan, right_university_raw, right_univ_raw_clean_withparan, token_n FROM all_dedup_matches) AS a 
                LEFT JOIN (
                    SELECT token AS left_token, token_rank AS left_token_rank, token_freq AS left_token_freq FROM all_token_freqs
                ) AS b ON a.left_token = b.left_token
                LEFT JOIN (
                    SELECT token AS right_token, token_rank AS right_token_rank, token_freq AS right_token_freq FROM all_token_freqs
                ) AS c ON a.right_token = c.right_token
            )
        )
    ) WHERE token_rank = 1
    """)

    dedup_filt = con.sql("SELECT * FROM dedup_res WHERE (min_freq AND (max_jaro_sim OR max_dl_sim)) OR namecontain")

    con.sql(f"COPY match_filt TO '{openalex_match_filt_file}'")
    con.sql(f"COPY dedup_filt TO '{dedup_match_filt_file}'")
    con.sql(f"COPY all_token_freqs TO '{all_token_freqs_file}'")
    log(f"Saved tokenmatch outputs: {openalex_match_filt_file}, {dedup_match_filt_file}, {all_token_freqs_file}")
else:
    log("Skipping token-level OpenAlex/dedup matching (tokenmatch=False); will read saved files")

################################
## MATCHING TO LOCATION TOKENS ##
################################
# tokenizing geonames
geonames_tokens_bycountry = con.sql(f""" 
    SELECT token, countryname, n_token_country/(SUM(n_token_country) OVER(PARTITION BY token)) AS country_freq, (SUM(n_token_country) OVER(PARTITION BY token))/(SUM(n_token_country) OVER()) AS token_freq, n_token_country, SUM(n_token_country) OVER(PARTITION BY token) AS n_token FROM (
        SELECT token, countryname, COUNT(*) AS n_token_country FROM (
            {help.tokenize_sql(col = 'name_clean',
                               id = 'geonameid',
                               tab = "(SELECT * FROM geonames WHERE type != 'city_altname')", 
                               othercols = ",type, countryname, name_clean")}
        ) GROUP BY token, countryname
    )
""")

#sorting by total frequency (note: this upweights countries with many geographies)
geonames_tokens_bytoken = con.sql(f""" 
    SELECT *, n_token/(SUM(n_token) OVER()) AS token_freq, ROW_NUMBER() OVER(ORDER BY n_token DESC) AS token_rank FROM (
        SELECT token, 
            list_transform(list_zip(
                    ARRAY_AGG(countryname ORDER BY n_token_country DESC),
                    ARRAY_AGG(n_token_country ORDER BY n_token_country DESC)
                ), x -> struct_pack(countryname := x[1], count := x[2])) AS countries,
            ARRAY_AGG(countryname ORDER BY n_token_country DESC)[1] AS top_country,
            ARRAY_AGG(n_token_country ORDER BY n_token_country DESC)[1]/SUM(n_token_country) AS top_country_freq,
            SUM(n_token_country) AS n_token FROM (
        SELECT *, SUM(n_token_country) OVER(PARTITION BY countryname) AS n_country, n_token_country/(SUM(n_token_country) OVER(PARTITION BY countryname)) AS country_rel_freq FROM (                        
            SELECT token, countryname, COUNT(*) AS n_token_country FROM (
                {help.tokenize_sql(col = 'name_clean',
                                id = 'geonameid',
                                tab = "(SELECT * FROM geonames WHERE type != 'city_altname')", 
                                othercols = ",type, countryname, name_clean")}
            ) GROUP BY token, countryname
        )
    ) GROUP BY token)
""")

rev_tokenized_forgeo = con.sql(f"SELECT * FROM ({help.tokenize_sql(col = 'univ_raw_clean_withparan', 
                                                   id = 'univ_id',
                                                   tab = "(SELECT * FROM univ_names_formatch WHERE matchind = 'nomatch')",
                                                   othercols = ', univ_raw_clean_withparan, university_raw')}) AS a LEFT JOIN all_token1_freqs AS b ON a.token = b.token")

# merging
if tokenmatch:
    print('merging geographies', flush=True)
    t0 = time.time()
    geo_merge = con.sql("SELECT a.token AS rev_token, b.token AS geo_token FROM (SELECT * FROM all_token1_freqs WHERE token_rank > 50) AS a JOIN (SELECT * FROM geonames_tokens_bytoken WHERE token_rank > 50 OR top_country_freq = 1) AS b ON jaro_similarity(a.token, b.token) >= 0.95")

    con.sql(f"COPY geo_merge TO '{geonames_token_merge_file}'")
    t1 = time.time()
    print(f"done! time: {t1-t0}s", flush=True)

################################
## COMBINING ALL MATCHES ##
################################
# good matches (exact, geoexact, revelio with >= 20 n_top_country)
good_matches = con.sql(
"""
SELECT university_raw, match_country FROM (
    SELECT *, COUNT(DISTINCT match_country) OVER(PARTITION BY university_raw) AS n_countries FROM (
        SELECT *, CASE WHEN priority = (MAX(priority) OVER(PARTITION BY university_raw)) THEN 1 ELSE 0 END AS keep FROM (
            SELECT university_raw, match_country, 3 AS priority, matchtype FROM exactmatches WHERE match_country IS NOT NULL AND match_country != 'NA' GROUP BY university_raw, match_country, matchtype 
            UNION ALL
            SELECT university_raw, 'United States' AS match_country, 2 AS priority, 'exact_ipeds' AS matchtype FROM ipeds_us_matches
            UNION ALL 
            SELECT university_raw, get_std_country(countryname) AS match_country, CASE WHEN match_score = 1 THEN 1 ELSE 0 END AS priority, 'exact_geo' AS matchtype FROM exact_geomatches
            UNION ALL 
            SELECT university_raw, get_std_country(top_country) AS match_country, 0 AS priority, 'exact_revelio' AS matchtype FROM univ_names WHERE top_country IS NOT NULL AND top_country_n_users >= 20
        ) 
    ) WHERE keep = 1
) GROUP BY university_raw, match_country
"""
)
log(f"Computed good_matches: {good_matches.shape[0]} rows")

# other matches (token, dedup, geo, gmaps)
legacy_openalex_match_filt_file = rcfg.OPENALEX_MATCH_FILT_PARQUET_LEGACY
legacy_dedup_match_filt_file = rcfg.DEDUP_MATCH_FILT_PARQUET_LEGACY

if os.path.exists(openalex_match_filt_file):
    log(f"Reading token match file: {openalex_match_filt_file}")
    match_filt = con.read_parquet(openalex_match_filt_file)
else:
    log(f"Missing {openalex_match_filt_file}; using legacy {legacy_openalex_match_filt_file}")
    match_filt = con.read_parquet(legacy_openalex_match_filt_file)

if os.path.exists(dedup_match_filt_file):
    log(f"Reading dedup match file: {dedup_match_filt_file}")
    dedup_filt = con.read_parquet(dedup_match_filt_file)
else:
    log(f"Missing {dedup_match_filt_file}; using legacy {legacy_dedup_match_filt_file}")
    dedup_filt = con.read_parquet(legacy_dedup_match_filt_file)

other_matches = con.sql(
"""
SELECT matches.university_raw, matchname_raw, match_country, matchscore, matchtype FROM (
    (SELECT university_raw, name AS matchname_raw, country_clean AS match_country, 
            (log_token_freq/(MIN(log_token_freq) OVER()))*(jaro_sim)*(CASE WHEN namecontain = 1 THEN 1 ELSE 0.7 END) AS matchscore, 
            'token' || token_n AS matchtype 
        FROM match_filt 
        UNION ALL 
        SELECT a.left_university_raw, a.right_university_raw AS matchname_raw, b.match_country, 
            (log_token_freq/(MIN(log_token_freq) OVER()))*(jaro_sim)*(CASE WHEN namecontain = 1 THEN 1 ELSE 0.7 END) AS matchscore, 
            'dedup_token' || token_n AS matchtype  
        FROM dedup_filt AS a 
        JOIN good_matches AS b 
        ON a.right_university_raw = b.university_raw
        UNION ALL
        SELECT university_raw, token AS matchname_raw, countryname AS match_country, 
            CASE WHEN match_score > 0.1 THEN score ELSE score*0.5 END AS matchscore, 'geomatch' AS matchtype
        FROM all_geomatches
        UNION ALL
        (SELECT b.university_raw, gmaps_name AS matchname_raw, get_std_country(univ_gmaps_country) AS match_country, 
            jaro_similarity(lower(b.university_raw), lower(gmaps_name)) AS matchscore, 'gmaps' AS matchtype 
        FROM gmaps_clean AS a LEFT JOIN univ_names AS b ON a.university_raw = lower(b.university_raw) WHERE a.university_raw IS NOT NULL AND univ_gmaps_country != 'No valid country match found' AND univ_gmaps_country IS NOT NULL)
        UNION ALL
        (SELECT university_raw, university_raw AS matchname_raw, top_country AS match_country, 0 AS matchscore, 'revelio' AS matchtype FROM univ_names WHERE top_country IS NOT NULL AND top_country_n_users < 20)
    ) AS matches
    LEFT JOIN
    (SELECT university_raw, 1 AS goodmatchind FROM good_matches GROUP BY university_raw) AS univs
    ON matches.university_raw = univs.university_raw 
) WHERE goodmatchind IS NULL
"""
)

# combining all matches
## TODO: actually do something with the scores?
final_matches = con.sql(
"""
    SELECT university_raw, match_country, 1 AS matchscore, 'exact' AS matchtype FROM good_matches 
    UNION ALL
    SELECT university_raw, match_country, matchscore, matchtype FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY university_raw, match_country ORDER BY matchscore DESC) AS rn FROM other_matches) WHERE rn = 1
""")
log(f"Computed final_matches: {final_matches.shape[0]} rows")

con.sql(f"COPY final_matches TO '{final_inst_country_file}'")
log(f"Saved: {final_inst_country_file}")
log(f"Done. Total elapsed: {round((time.time() - t_script0)/60, 2)} min")

# # random sample of ids
# ids = "','".join(con.sql(help.random_ids_sql('university_raw','other_matches', n = 100)).df()['university_raw'].astype(str))

# con.sql(f"""SELECT university_raw, match_country, matchscore, matchtype FROM other_matches WHERE university_raw IN ('{ids}') ORDER BY university_raw, matchscore DESC""").df()

# # combining all openalex matches
# # TODO: finish -- i think we want to pivot long and join on tokens, for each potential match take the min frequency of matched tokens; average frequency; n matched tokens; jaro winkler; length of overlapping sequence
# match_filt = con.read_parquet(f'{root}/data/int/rev_openalex_match_filt_jun20.parquet')

# openalex_matches_all = con.sql("SELECT openalexid AS matchid, university_raw, name AS matchname_raw, univ_text_clean_corr AS rev_clean, name_clean AS matchname_clean, country_clean AS match_country, 1 AS matchscore, 'exact' AS matchtype FROM allmatches UNION ALL (SELECT openalexid, university_raw, name, univ_raw_clean_withparan AS rev_clean, name_clean, country_clean AS match_country, (log_token_freq/(MIN(log_token_freq) OVER()))*(jaro_sim)*(CASE WHEN namecontain = 1 THEN 1 ELSE 0.7 END) AS matchscore, 'token' || token_n AS matchtype FROM match_filt)")


# # combining all geo matches -- FOR NOW, JUST USING all_geomatches (no fuzzy matching on geo)
# # geo_merge = con.read_parquet(f'{root}/data/int/rev_geonames_token_merge_jun20.parquet')

# # geotokenmerge = con.sql(
# # """
# # SELECT geo_token, rev_token, token_match, idnum AS univ_id, university_raw, countryname, country_freq, token_freq, n_token_country, n_token FROM (
# #     SELECT * FROM (
# #         SELECT token AS rev_token, token_1 AS geo_token, CASE WHEN rev_token = geo_token THEN 1 ELSE 0 END AS token_match, MAX(CASE WHEN rev_token = geo_token THEN 1 ELSE 0 END) OVER(PARTITION BY token) AS max_tokenmatch
# #         FROM geo_merge  
# #     ) WHERE token_match = max_tokenmatch) AS g
# #     LEFT JOIN geonames_tokens_bycountry AS geo 
# #     ON g.geo_token = geo.token
# #     LEFT JOIN rev_tokens_withparan AS rev
# #     ON g.rev_token = rev.token
# # """
# # )

# # geo_matches_all = con.sql(
# # """
# # SELECT univ_id, NULL AS matchid, university_raw, NULL AS matchname_raw, rev_token AS rev_clean, geo_token AS matchname_clean, countryname AS match_country, country_freq AS matchscore, 'geotoken' AS matchtype FROM geotokenmerge
# # UNION ALL
# # (SELECT univ_id, geonameid AS matchid, university_raw, NULL AS matchname_raw, token AS rev_clean, token AS matchname_clean, countryname, match_score_max AS matchscore, 'citymatch' AS matchtype FROM citymatch_final)
# # """
# # )

# # all matches
# matches_combined = con.sql(
# """
# SELECT *, CASE WHEN matchtype = 'exact' OR matchtype = 'exact_geomatch' OR (matchtype = 'revelio' AND matchscore >= 20) THEN 1 ELSE 0 END AS goodmatchind, MAX(CASE WHEN matchtype = 'exact' OR matchtype = 'exact_geomatch' OR (matchtype = 'revelio' AND matchscore >= 20) THEN 1 ELSE 0 END) OVER(PARTITION BY university_raw) AS max_goodmatchind FROM (
#     SELECT * FROM openalex_matches_all
#     UNION ALL 
#     SELECT univ_id, geonameid AS matchid, university_raw, NULL AS matchname_raw, token AS rev_clean, token AS matchname_clean, countryname AS match_country, 
#         CASE WHEN match_score >= 0.5 AND ncountry = 1 THEN match_score 
#             WHEN match_score > 0.1 THEN 1 + score
#             ELSE score END AS match_score, 
#         CASE WHEN match_score >= 0.5 AND ncountry = 1 THEN 'exact_geomatch' ELSE 'geomatch' END AS matchtype FROM all_geomatches
#     UNION ALL
#     (SELECT univ_id, NULL AS matchid, b.university_raw, gmaps_name AS matchname_raw, a.university_raw AS rev_clean, gmaps_name AS matchname_clean, univ_gmaps_country AS match_country, NULL AS matchscore, 'gmaps' AS matchtype FROM gmaps_clean AS a LEFT JOIN univ_names AS b ON a.university_raw = lower(b.university_raw) WHERE a.university_raw IS NOT NULL)
#     UNION ALL
#     (SELECT univ_id, NULL AS matchid, university_raw, NULL AS matchname_raw, university_raw AS rev_clean, NULL AS matchname_clean, top_country AS match_country, top_country_n_users AS matchscore, 'revelio' AS matchtype FROM univ_names WHERE top_country IS NOT NULL)
# )
# """
# )

# # good matches
# matches_good = con.sql("SELECT * FROM matches_combined WHERE goodmatchind = 1")

# # other matches, joining to dedup
# dedup_raw = con.read_parquet(f'{root}/data/int/rev_dedup_match_jun20.parquet')
# matches_with_dedup = con.sql(
# """
# SELECT * FROM (
#     (SELECT university_raw FROM matches_combined WHERE max_goodmatchind = 0 GROUP BY university_raw) AS a
#     LEFT JOIN dedup_raw AS b
#     ON a.university_raw = b.left_university_raw
#     LEFT JOIN (SELECT university_raw, match_country, MAX(CASE WHEN matchtype = 'exact' THEN 1 ELSE 0 END) AS exactind, MAX(CASE WHEN matchtype = 'exact_geomatch' THEN 1 ELSE 0 END) AS exactgeomatchind, MAX(CASE WHEN matchtype = 'revelio' THEN 1 ELSE 0 END) AS revelioind FROM matches_good GROUP BY university_raw, match_country) AS c
#     ON b.right_university_raw = c.university_raw
# )
# """)


# con.sql(f"""SELECT university_raw, matchtype, matchname_clean, match_country, matchscore, * FROM matches_combined WHERE university_raw IN ('{ids}') ORDER BY university_raw""").df()

# # want something like -- for each category of match, 'strength' of match (within category) and match country


# matches_combined = con.sql(
# """
# SELECT * FROM 
#     (SELECT univ_id, university_raw,  FROM allmatches)
#     UNION ALL
#     ()
# """
# )


# ## OLD!
# # three-token tokenization
# samp_size = 1000
# con.sql(f"CREATE OR REPLACE TABLE rev_tokenized AS ({help.tokenize_nword_sql(col = 'univ_raw_clean_withparan', 
#                                                    id = 'univ_id',
#                                                    tab = '(SELECT * FROM univ_names_formatch WHERE exactmatchind = 0)',
#                                                    n = 3,
#                                                    othercols = ', univ_raw_clean_withparan, university_raw')})")

# con.sql(f"CREATE OR REPLACE TABLE openalex_tokenized AS ({help.tokenize_nword_sql(col = 'name_clean',
#                                                      id = 'inst_id',
#                                                      tab = 'inst_clean',
#                                                      n = 3,
#                                                      othercols = ', name_clean, name')})")

# # token frequencies 
# token_freqs = con.sql("SELECT token, COUNT(*)/MEAN(n) AS token_freq, ROW_NUMBER() OVER(ORDER BY token_freq DESC) AS token_rank FROM (SELECT token, COUNT(*) OVER() AS n FROM (SELECT token FROM rev_tokenized UNION ALL SELECT token FROM openalex_tokenized)) GROUP BY token ORDER by token_freq DESC")

# # taking samples
# con.sql(f"CREATE OR REPLACE TABLE revsamp AS (SELECT * FROM rev_tokenized ORDER BY RANDOM() LIMIT {samp_size})")
# con.sql(f"CREATE OR REPLACE TABLE openalexsamp AS (SELECT * FROM openalex_tokenized ORDER BY RANDOM() LIMIT {samp_size})")

# # matching
# token3_match = con.sql("SELECT * FROM rev_tokenized AS a JOIN openalex_tokenized AS b ON a.token = b.token")

# token3_match_fuzzy = con.sql(f"SELECT * FROM (SELECT * FROM revsamp) AS a JOIN (SELECT * FROM openalex_tokenized) AS b ON damerau_levenshtein(a.token, b.token) < 0.9")


# # tokenizing revelio and merging to get freqs + spell-correcting
# rev_tokenized = con.sql(
#     """SELECT university_raw, top_country, n_users, univ_id, c.token AS originaltoken, rev_token_id, c.token_freq, c.token_rank AS originaltoken_rank, 
#     (CASE WHEN token_corrected IS NOT NULL THEN token_corrected ELSE c.token END) AS token, 
#     (CASE WHEN token_corrected_rank IS NOT NULL THEN token_corrected_rank ELSE c.token_rank END) AS token_rank,
#     (CASE WHEN token_corrected IS NOT NULL THEN 1 ELSE openalex END) AS openalex,
#     FROM (
#         (SELECT *, 
#             unnest(regexp_split_to_array(univ_raw_clean_withparan, ' ')) AS token, 
#             generate_subscripts(regexp_split_to_array(univ_raw_clean_withparan, ' '), 1) AS rev_token_id 
#         FROM univ_names) AS a 
#         LEFT JOIN all_token_freqs AS b 
#         ON a.token = b.token
#     ) AS c 
#     LEFT JOIN spell_cw_jw_all AS d 
#     ON c.token = d.token
#     """)

# # tokenizing openalex
# openalex_tokenized = con.sql("SELECT * FROM ((SELECT *, unnest(regexp_split_to_array(name_clean, ' ')) AS token, generate_subscripts(regexp_split_to_array(name_clean, ' '), 1) AS openalex_token_id FROM inst_clean) AS a LEFT JOIN all_token_freqs AS b ON a.token = b.token)")

# # merging!
# rank_cutoff = 1000
# merged_tokens_raw = con.sql(f"SELECT university_raw, top_country, n_users, univ_id, originaltoken, rev_token_id, a.token_freq, originaltoken_rank, a.token_rank, a.token, name, inst_id, openalexid, country_clean, openalex_token_id FROM ((SELECT * FROM rev_tokenized WHERE openalex = 1 AND token_rank > {rank_cutoff}) AS a LEFT JOIN (SELECT * FROM openalex_tokenized WHERE token_rank > {rank_cutoff}) AS b ON a.token = b.token)")

# # filtering
# merged_tokens = con.sql(
# """
#     SELECT * FROM(
#         SELECT university_raw, name, univ_id, inst_id, 
#             list_transform(list_zip(array_agg(token ORDER BY rev_token_id),
#                                     array_agg(token_freq ORDER BY rev_token_id)),
#                                     x -> struct_pack(token := x[1], freq := x[2])
#                             ) AS token_list,
#                             COUNT(*) AS n_tokens
#         FROM merged_tokens_raw 
#         GROUP BY university_raw, name, univ_id, inst_id
#     )
# """)


# # testing
# ntot = con.sql("SELECT COUNT(DISTINCT univ_id) FROM rev_tokenized ").df().iloc[0,0]

# #testdfs = []
# # testing shapes of diff cutoffs
# for cutoff in [50, 100, 250, 500, 1000]:
#     print(f"Cutoff rank: {cutoff}")
#     # universe of possible matches
#     n = con.sql(f"SELECT COUNT(DISTINCT univ_id) FROM rev_tokenized WHERE openalex = 1 AND token_rank > {cutoff}").df().iloc[0,0]
#     print(f"Total number of institutions to be matched: {n} ({round(100*n/ntot, 2)}% of total)")

#     merged_tokens_temp = con.sql(f"SELECT university_raw, top_country, n_users, univ_id, originaltoken, rev_token_id, a.token_freq, originaltoken_rank, a.token_rank, a.token, name, openalexid, country_clean, openalex_token_id FROM ((SELECT * FROM rev_tokenized WHERE openalex = 1 AND token_rank > {cutoff}) AS a LEFT JOIN (SELECT * FROM openalex_tokenized WHERE token_rank > {cutoff}) AS b ON a.token = b.token)")

#     print(f"Total number of matches: {merged_tokens_temp.shape}")
#     print(f"Total number of unique matches: {con.sql("SELECT univ_id, openalexid FROM merged_tokens_temp GROUP BY univ_id, openalexid").shape}")

#     # testdfs.append(con.sql("SELECT token, university_raw, name, token_rank, top_country, country_clean FROM merged_tokens_temp ORDER BY RANDOM() LIMIT 100").df())
    


# ## OLD:
# cleanmatches = con.sql(
# """
#     SELECT *, COUNT(*) OVER(PARTITION BY univ_id) AS nmatch FROM 
#         ((SELECT * FROM univ_names_spellcorr WHERE LENGTH(univ_raw_clean) > 2) AS a 
#         JOIN (SELECT * FROM inst_clean WHERE LENGTH(name_clean) > 2 AND type != 'company') AS b 
#         ON a.univ_raw_clean = b.name_clean)""")

# correctedmatches 

# con.sql("SELECT university_raw, name, top_country, country_clean, nmatch FROM exactmatchesclean WHERE nmatch = 1 AND (top_country IS NULL OR country_clean = 'NA' OR top_country = country_clean) AND (LENGTH(name_clean) > 5 OR top_country = country_clean)").df().sample(1000)


# #### TESTING JOINS
# from rapidfuzz import fuzz, utils, process
# import pandas as pd 
# import time 

# t0 = time.time()
# n = 10000
# allmatchesdf = allmatches.df()
# univ_ids = np.random.permutation(allmatchesdf['univ_id'].unique())[:n]
# inst_ids = allmatchesdf.loc[allmatchesdf['univ_id'].isin(univ_ids)]['openalexid'].unique()

# univ_namesdf = con.sql("SELECT * FROM univ_names").df()
# univ_ids_altsamp = np.random.permutation(univ_namesdf['univ_id'].unique())[:n]
# univ_samp = univ_namesdf.loc[univ_namesdf['univ_id'].isin(np.concatenate([univ_ids, univ_ids_altsamp]))]

# inst_cleandf = inst_clean.df()
# inst_ids_altsamp = np.random.permutation(inst_cleandf['openalexid'].unique())[:n]
# inst_samp = inst_cleandf.loc[inst_cleandf['openalexid'].isin(np.concatenate([inst_ids, inst_ids_altsamp]))]

# univ_test = univ_samp['university_raw']
# inst_test = inst_samp['name']

# con.sql("SELECT * FROM inst_samp").to_csv(f"{root}/data/int/inst_samp.csv")
# con.sql("SELECT * FROM univ_samp").to_csv(f"{root}/data/int/univ_samp.csv")


# t1 = time.time()
# print(f"Time for processing: {t1-t0}s")

# matches = pd.DataFrame(process.cdist(univ_test, inst_test, scorer = fuzz.token_sort_ratio, processor = utils.default_process))
# matches.columns = ['matchscore' + str(id) for id in inst_samp['inst_id']]
# matches['univ_id'] = univ_samp.reset_index()['univ_id']

# matchlong = pd.wide_to_long(matches, 'matchscore', i ='univ_id', j = 'inst_id').reset_index()
# matchlong_merge = matchlong.loc[matchlong['matchscore']>=85].merge(univ_samp[['university_raw', 'univ_id']], 'left', on = 'univ_id').merge(inst_samp[['name','inst_id']], 'left', on = 'inst_id')
# matchlong_merge[['university_raw','name','matchscore']]

# t2 = time.time()
# print(f"Time for rapidfuzz: {t2-t1}s")
