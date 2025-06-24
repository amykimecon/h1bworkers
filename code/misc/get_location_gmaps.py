# File Description: Getting Google Maps Locations from School Names
# Author: Amy Kim
# Date Created: Tue Apr 8

# Imports and Paths
import duckdb as ddb
import pandas as pd
import re
import requests
import random
import time
import os
from datetime import datetime
import pyreadr
import json
from requests.exceptions import ConnectionError

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"
con = ddb.connect()

API_KEY = 'AIzaSyCk4kmzWdd5VML64UnTVKvEzmb9d93Qm7A'

# returns sql string for cleaned institution name given column name of institution
def inst_clean_regex_sql(col):
    str_out = f"""
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    -- strip accents and convert to lowercase
                    strip_accents(lower({col})), 
                -- remove anything in parantheses
                '\\s*(\\(|\\[).*(\\)|\\])\\s*', ' ', 'g'), 
            -- remove apostrophes, periods
            $$'|’|\\.$$, '', 'g'),
        -- convert 'and' symbols to text
        '\\s?&\\s?', ' and ', 'g'), 
    -- remove any other non alphanumeric symbols and replace w space
    '[^a-z0-9\\s]', ' ', 'g')
    """

    return f"TRIM(REGEXP_REPLACE({str_out}, '\\s+', ' ', 'g'))" 

# same as above but keeps text in parentheses 
def inst_clean_withparan_regex_sql(col):
    str_out = f"""
    REGEXP_REPLACE(
        REGEXP_REPLACE(
                REGEXP_REPLACE(
                    -- strip accents and convert to lowercase
                    strip_accents(lower({col})), 
            -- remove apostrophes, periods
            $$'|’|\\.$$, '', 'g'),
        -- convert 'and' symbols to text
        '\\s?&\\s?', ' and ', 'g'), 
    -- remove any other non alphanumeric symbols and replace w space
    '[^a-z0-9\\s]', ' ', 'g')
    """

    return f"TRIM(REGEXP_REPLACE({str_out}, '\\s+', ' ', 'g'))" 

def get_gmaps_json_from_school_name(school_name, apikey = API_KEY, retries = 5):
    if pd.isnull(school_name):
        return None
    
    if re.search("school", school_name) is None:
        school_name = school_name + " school"
    
    for attempt in range(retries):
        try:
            # Google Places API URL
            url = f"""https://maps.googleapis.com/maps/api/place/findplacefromtext/json?fields=formatted_address%2Cname&input={re.sub("\\s", "%20", school_name)}&inputtype=textquery&key={apikey}"""
    
            # Send request to Google Places API
            response = requests.get(url)
            #print(response.json())
            response.raise_for_status()
            return response.json()
        except (ConnectionError, requests.exceptions.RequestException) as e:
            wait = (2 ** attempt) + random.random()
            print(f"Retrying in {wait:.2f}s due to error: {e}")
            time.sleep(wait)

    return None 

# Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

def get_std_country(country, dict = country_cw_dict):
    if country is None:
        return None 
    
    if country in dict.keys():
        return dict[country]
    
    if country in dict.values():
        return country 
    
    return "No Country Match"

con.create_function("get_std_country", lambda x: get_std_country(x), ['VARCHAR'], 'VARCHAR')


# Importing Data (From WRDS Server)
rev_raw = con.read_parquet(f"{root}/data/wrds/wrds_out/rev_user_merge0.parquet")

for j in range(1,10):
    rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{root}/data/wrds/wrds_out/rev_user_merge{j}.parquet'")
    print(rev_raw.shape)

# Importing institutions data
institutions = con.read_csv(f"{root}/data/crosswalks/institutions.csv")
acronyms = con.read_csv(f"{root}/data/crosswalks/institutions_acronyms.csv")
altnames = con.read_csv(f"{root}/data/crosswalks/institutions_altnames.csv")

# Cleaning Info
rev_clean = con.sql(
f"""
    SELECT 
    fullname, university_country, university_location, degree, user_id,
    CASE 
        WHEN lower(university_raw) ~ '.*(high\\s?school).*' OR (degree = '<NA>' AND university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?)$') THEN 'High School' 
        WHEN degree != '<NA>' THEN degree
        WHEN lower(degree_raw) ~ '.*(cert|credential|course|semester|exchange|abroad|summer|internship|edx|cdl|coursera).*' OR lower(university_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera).*' OR lower(field_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera).*' THEN 'Non-Degree'
        WHEN (lower(degree_raw) ~ '.*(undergrad).*') OR (degree_raw ~ '.*(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?|B\\.?I\\.?S\\.?).*') OR degree_raw ~ '^B\\.?\\s?S\\.?.*' OR lower(field_raw) ~ '.*bachelor.*' OR lower(degree_raw) ~ '.*bachelor.*' THEN 'Bachelor'
        WHEN lower(degree_raw) ~ '.*(master).*' OR degree_raw ~ '^M\\.?(Eng|Sc|A)\\.?.*' THEN 'Master'
        WHEN degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?|M\\.?Eng|M\\.?Sc|M\\.?A).*' OR lower(field_raw) ~ '.*master.*' OR lower(degree_raw) ~ '.*master.*' THEN 'Master'
        WHEN lower(field_raw) ~ '.*(associate).*' OR degree_raw ~ 'A\\.?\\s?A\\.?.*' THEN 'Associate' 
        WHEN degree_raw ~ '^B\\.?\\s?[A-Z].*' THEN 'Bachelor'
        WHEN degree_raw ~ '^M\\.?\\s?[A-Z].*' THEN 'Master'
        ELSE degree END AS degree_clean,
    {inst_clean_regex_sql('university_raw')} AS univ_raw_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw
"""
)

# grouping by user x university (filtering out non-degree) x country, then by university x country, then by university (taking top non-null country)
con.sql(
"""CREATE OR REPLACE TABLE univ_names AS (SELECT university_raw, univ_raw_clean,
    (ARRAY_AGG(university_country ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country, 
    (ARRAY_AGG(n_users_univ_ctry ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country_n_users, 
    SUM(n_users_univ_ctry) AS n_users,    
    SUM(n_hs_ctry)/SUM(n_users_univ_ctry) AS share_hs,
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

con.sql("SELECT * FROM univ_names").to_csv(f"{root}/data/int/rev_univ_names.csv")

con.sql(
"""CREATE OR REPLACE TABLE univ_names_nohs AS (SELECT university_raw, univ_raw_clean,
    (ARRAY_AGG(university_country ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country, 
    (ARRAY_AGG(n_users_univ_ctry ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country_n_users, 
    SUM(n_users_univ_ctry) AS n_users 
FROM (
    SELECT university_raw, univ_raw_clean, university_country, COUNT(*) AS n_users_univ_ctry 
    FROM (
        SELECT university_raw, univ_raw_clean, user_id, university_country 
        FROM rev_clean 
        WHERE degree_clean != 'Non-Degree' AND degree_clean != 'High School'
        GROUP BY university_raw, univ_raw_clean, university_country, user_id
    ) GROUP BY university_country, university_raw, univ_raw_clean
) GROUP BY university_raw, univ_raw_clean ORDER BY n_users DESC)
""")

con.sql("SELECT * FROM univ_names_nohs").to_csv(f"{root}/data/int/rev_univ_names_nohs.csv")

# cleaning openalex institutions
inst_clean = con.sql(
f"""
SELECT *, CASE WHEN country_code = 'NA' THEN 'NA' ELSE get_std_country(country_code) END AS country_clean, ROW_NUMBER() OVER() AS inst_id, {inst_clean_regex_sql('name')} AS name_clean FROM (
    SELECT * FROM (
        SELECT name, country_code, type, 'institutions' AS source FROM institutions
        UNION ALL 
        SELECT alternative_names AS name, country_code, type, 'altnames' AS source FROM altnames 
        UNION ALL
        SELECT acronyms AS name, country_code, type, 'acronyms' AS source FROM acronyms
    )
    GROUP BY name, country_code, type, source
)
""")

con.sql("SELECT * FROM inst_clean").to_csv(f"{root}/data/int/allinstitutions_clean.csv")

# tokenizing
all_names_tokenized = con.sql(
"""
    SELECT token, (COUNT(*) OVER(PARTITION BY token))/(COUNT(*) OVER()) AS token_freq FROM (SELECT
    unnest(regexp_split_to_array(univ_raw_clean, ' ')) AS token 
    FROM univ_names UNION ALL SELECT unnest(regexp_split_to_array(name_clean, ' ')) AS token FROM inst_clean) WHERE token != ''
""")

token_freqs = con.sql("SELECT token, MEAN(token_freq) AS token_freq FROM all_names_tokenized GROUP BY token").df()

freq_cutoff = token_freqs.sort_values('token_freq', ascending=False).iloc[500,1]

# univ_names_tokenized_bycountry = con.sql("SELECT token, top_country, COUNT(*) AS freq, SUM(n_users) AS tot_users, SUM(top_country_n_users) AS top_country_users, SUM(top_country_n_users)/MEAN(token_users_top_country) AS share_top_country_users_country, SUM(n_users)/MEAN(token_users_all) AS share_users_country FROM (SELECT *, SUM(top_country_n_users) OVER(PARTITION BY token) AS token_users_top_country, SUM(n_users) OVER(PARTITION BY token) AS token_users_all FROM univ_names_tokenized) WHERE top_country IS NOT NULL GROUP BY token, top_country")

inst_clean_tokenized = con.sql("SELECT name, country_code, country_clean, type, source, inst_id, name_clean, a.token AS token, token_freq, COUNT(*) OVER(PARTITION BY inst_id) AS inst_n_rare FROM ((SELECT *, unnest(regexp_split_to_array(name_clean,' ')) AS token FROM inst_clean) AS a LEFT JOIN token_freqs AS b ON a.token = b.token)")

## getting exact matches on university name (before cleaning)
# exact matches to openalex name, altname, acronym: include hs
exactmatches = con.sql("SELECT * FROM (univ_names AS a JOIN inst_clean AS b ON a.university_raw = b.name)")

exactmatchesclean = con.sql("""SELECT *, COUNT(*) OVER(PARTITION BY univ_id) AS nmatch FROM ((SELECT * FROM univ_names WHERE LENGTH(univ_raw_clean) > 2) AS a JOIN (SELECT * FROM inst_clean WHERE LENGTH(name_clean) > 2 AND type != 'company') AS b ON a.univ_raw_clean = b.name_clean)""")


con.sql("SELECT university_raw, name, top_country, country_clean, nmatch FROM exactmatchesclean WHERE nmatch = 1 AND (top_country IS NULL OR country_clean = 'NA' OR top_country = country_clean) AND (LENGTH(name_clean) > 5 OR top_country = country_clean)").df().sample(1000)

# remainder: exclude hs?
univ_names2 = con.sql("SELECT university_raw, univ_raw_clean, top_country, top_country_n_users, n_users, univ_id FROM (univ_names AS a LEFT JOIN inst_clean AS b ON a.university_raw = b.name) WHERE b.name IS NULL AND (top_country IS NULL OR top_country_n_users < 10)")

univ_names_noexactmatch = con.sql("SELECT university_raw, univ_raw_clean, top_country, top_country_n_users, n_users, a.univ_id FROM (univ_names AS a LEFT JOIN (SELECT univ_id FROM exactmatchesclean GROUP BY univ_id) AS b ON a.univ_id = b.univ_id) WHERE b.univ_id IS NULL")

univtest = con.sql("SELECT * FROM univ_names2 LIMIT 100000")

univ_names2_tokenized = con.sql("SELECT university_raw, univ_raw_clean, top_country, univ_id, a.token AS token, token_freq, COUNT(*) OVER (PARTITION BY univ_id) AS univ_n_rare FROM ((SELECT *, unnest(regexp_split_to_array(univ_raw_clean, ' ')) AS token FROM univ_names2) AS a LEFT JOIN token_freqs AS b ON a.token = b.token)")
    

## merging tokenized tables
tokenmatches = con.sql(
f"""
    SELECT a.inst_id, b.univ_id, a.name, a.name_clean, b.university_raw, b.univ_raw_clean, a.country_clean, b.top_country, COUNT(*) AS nmatch FROM (
        (SELECT * FROM inst_clean_tokenized WHERE token != '' AND token_freq < {freq_cutoff}) AS a JOIN (SELECT * FROM univ_names2_tokenized WHERE token != '' AND token_freq < {freq_cutoff}) AS b 
        ON a.token = b.token
    ) GROUP BY a.inst_id, b.univ_id, a.name, a.name_clean, b.university_raw, a.country_clean, b.top_country, b.univ_raw_clean
""")
con.sql("SELECT COUNT(*) FROM tokenmatches")

tokenmatches_filt = con.sql("SELECT * FROM tokenmatches WHERE (univ_raw_clean LIKE '% %' AND POSITION(univ_raw_clean IN name_clean) > 0) OR (name_clean LIKE '% %' AND POSITION(name_clean IN univ_raw_clean) > 0)")

tokenmatches_df = tokenmatches.df()

## looking for all partial matches on cleaned names
instmatches = con.sql(
"""
SELECT * FROM (
    SELECT *
    FROM univtest AS a 
    JOIN (SELECT * FROM inst_clean WHERE type = 'education' OR type = 'healthcare' OR type = 'other') AS b
    ON 
    -- first, check if openalex name in revelio string (raw)
        (a.university_raw LIKE '% %' AND b.name LIKE '% %' AND POSITION(b.name IN a.university_raw) > 0) OR
    -- second, check if revelio in openalex name (raw)
        (b.name LIKE '% %' AND POSITION(a.university_raw) > 0) OR POSITION(a.univ_raw_clean IN b.name_clean) > 0
)
""")

instmatches_df = instmatches.df()


univ_names_openalex_clean = con.sql("SELECT * FROM (univ_names AS a JOIN inst_clean AS b ON a.univ_raw_clean = b.name_clean)")

univ_names_openalex_all = con.sql("SELECT * FROM (univ_names AS a JOIN inst_clean AS b ON POSITION(b.name_clean IN a.univ_raw_clean) > 0)")

univ_names_nolookup = univ_names.loc[(univ_names['top_country'].isnull() == 0)&(univ_names['top_country_n_users'] >= 10)]
univ_names_nolookup['gmaps_json'] = {}
univ_names_nolookup.to_parquet(f"{root}/data/int/gmaps_univ_locations/gmaps_univs_nolookup.parquet")

univ_names_for_lookup = univ_names.loc[(univ_names['top_country'].isnull() == 1)|(univ_names['top_country_n_users'] < 10)]
print(f"Total universities to check: {univ_names_for_lookup.shape[0]}")

gmaps = con.sql(f"""SELECT * FROM read_parquet('{"')UNION ALL SELECT * FROM read_parquet('".join([f"{root}/data/int/gmaps_univ_locations/{f}" for f in os.listdir(f"{root}/data/int/gmaps_univ_locations/")])}')""")

# # note (apr 8): already read rows 0-9999 (gmaps_univs0_to_9999.parquet)
# univ_names_for_lookup_unread = univ_names_for_lookup.iloc[10000:]
# i = 0
# d = 100

# while d*(i+1) < univ_names_for_lookup.shape[0]:
#     print(f"{datetime.now()} Iteration Number {i}")
#     tempdf = univ_names_for_lookup_unread.iloc[d*i:d*(i+1)].copy()
#     tempdf['gmaps_json'] = tempdf['university_raw'].apply(lambda x: get_gmaps_json_from_school_name(x))
#     tempdf.to_parquet(f"{root}/data/int/gmaps_univ_locations/gmaps_univs{i}.parquet")
#     i += 1

# tempdf = univ_names_for_lookup_unread.iloc[d*i:].copy()
# tempdf['gmaps_json'] = tempdf['university_raw'].apply(lambda x: get_gmaps_json_from_school_name(x))
# tempdf.to_parquet(f"{root}/data/int/gmaps_univ_locations/gmaps_univs{i}.parquet")