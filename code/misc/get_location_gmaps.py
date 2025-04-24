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
from datetime import datetime
from requests.exceptions import ConnectionError

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

API_KEY = 'AIzaSyCk4kmzWdd5VML64UnTVKvEzmb9d93Qm7A'

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

# Importing Data (From WRDS Server)
rev_raw = con.read_parquet(f"{root}/data/wrds/wrds_out/rev_user_merge0.parquet")

for j in range(1,10):
    rev_raw = con.sql(f"SELECT * FROM rev_raw UNION ALL SELECT * FROM '{root}/data/wrds/wrds_out/rev_user_merge{j}.parquet'")
    print(rev_raw.shape)

rev_clean = con.sql(
"""
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
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(strip_accents(lower(university_raw)), '\\s*\\(.*\\)\\s*', ' ', 'g'), '[^a-z0-9\\s]', ' ', 'g'), '\\s+', ' ', 'g')) AS univ_raw_clean,
    degree_raw, field_raw, university_raw
    FROM rev_raw
"""
)

# grouping by user x university (filtering out non-degree) x country, then by university x country, then by university (taking top non-null country)
univ_names = con.sql(
"""SELECT university_raw, 
    (ARRAY_AGG(university_country ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country, 
    (ARRAY_AGG(n_users_univ_ctry ORDER BY n_users_univ_ctry DESC) FILTER (WHERE university_country IS NOT NULL))[1] AS top_country_n_users, 
    SUM(n_users_univ_ctry) AS n_users 
FROM (
    SELECT university_raw, university_country, COUNT(*) AS n_users_univ_ctry 
    FROM (
        SELECT lower(university_raw) AS university_raw, user_id, university_country 
        FROM rev_clean 
        WHERE degree_clean != 'Non-Degree'
        GROUP BY university_raw, university_country, user_id
    ) GROUP BY university_country, university_raw
) GROUP BY university_raw ORDER BY n_users DESC
""").df()

univ_names_nolookup = univ_names.loc[(univ_names['top_country'].isnull() == 0)&(univ_names['top_country_n_users'] >= 10)]
univ_names_nolookup['gmaps_json'] = {}
univ_names_nolookup.to_parquet(f"{root}/data/int/gmaps_univ_locations/gmaps_univs_nolookup.parquet")

univ_names_for_lookup = univ_names.loc[(univ_names['top_country'].isnull() == 1)|(univ_names['top_country_n_users'] < 10)]
print(f"Total universities to check: {univ_names_for_lookup.shape[0]}")

# note (apr 8): already read rows 0-9999 (gmaps_univs0_to_9999.parquet)
univ_names_for_lookup_unread = univ_names_for_lookup.iloc[10000:]
i = 0
d = 100

while d*(i+1) < univ_names_for_lookup.shape[0]:
    print(f"{datetime.now()} Iteration Number {i}")
    tempdf = univ_names_for_lookup_unread.iloc[d*i:d*(i+1)].copy()
    tempdf['gmaps_json'] = tempdf['university_raw'].apply(lambda x: get_gmaps_json_from_school_name(x))
    tempdf.to_parquet(f"{root}/data/int/gmaps_univ_locations/gmaps_univs{i}.parquet")
    i += 1

tempdf = univ_names_for_lookup_unread.iloc[d*i:].copy()
tempdf['gmaps_json'] = tempdf['university_raw'].apply(lambda x: get_gmaps_json_from_school_name(x))
tempdf.to_parquet(f"{root}/data/int/gmaps_univ_locations/gmaps_univs{i}.parquet")