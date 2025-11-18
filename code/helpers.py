## HELPERS FOR CLEANING INDIVIDUAL REVELIO DATA
import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import jellyfish
import numpy as np
import pandas as pd
import tqdm
from rapidfuzz import distance, process
import importlib.util

# local
from config import * 

# SQL QUERIES
# sql string to extract random list of n ids
def random_ids_sql(id, tab, n = 10):
    return f"SELECT {id} FROM {tab} ORDER BY RANDOM() LIMIT {n}"

# returns sql string for tokenizing column 'col' with id column 'id' from table 'tab' and separator sep (default is space)
def tokenize_sql(col, id, tab, sep = "' '", othercols = ""):
    str_out = f"""
        SELECT
            unnest(regexp_split_to_array({col}, {sep})) AS token, 
            generate_subscripts(regexp_split_to_array({col}, {sep}), 1) AS token_id,
            {id} AS idnum {othercols}
        FROM {tab} WHERE {col} != ''
    """
    return str_out

# returns sql string for n-word tokenization of column 'col' with id column 'id' from table 'tab' and separator sep (default is space)
def tokenize_nword_sql(col, id, tab, n = 2, sep = "' '", othercols = ""):
    # start with one-word tokenization
    token1 = tokenize_sql(col, id, tab, sep, othercols)

    # on each iteration, update token_maintable to contain i-word tokenization
    i = 1
    tokentable = token1 
    
    # repeat until i = n
    while i < n:
        # joining i-tokenized table on one-word tokenized table repeatedly until i = n
        tokentable = f"""
            SELECT a.idnum, a.token_id, a.token || {sep} || b.token AS token {othercols}
            FROM (
                (SELECT * FROM (SELECT token, token_id, idnum, COUNT(token) OVER(PARTITION BY idnum) AS n_tokens {othercols} FROM ({tokentable})) WHERE token_id != n_tokens) AS a
                JOIN (SELECT token, token_id, idnum FROM ({token1}) WHERE token_id - {i} > 0) AS b
                ON a.token_id = b.token_id - {i} AND a.idnum = b.idnum
            )
        """
        i += 1

    return tokentable

# returns sql string for spell correction given crosswalk table (table containing spelling corrections for tokens), main table, string column in main table to be corrected
def spell_corr_sql(cw_tab, main_tab, col, id):
    str_out = f"""
        SELECT * FROM 
            (SELECT * FROM {main_tab} AS a 
            LEFT JOIN 
                (SELECT idnum, STRING_AGG(token_repl, ' ' ORDER BY token_id) AS {col}_corr FROM 
                    (SELECT *, CASE WHEN token_corrected IS NOT NULL THEN token_corrected ELSE a.token END AS token_repl FROM ({tokenize_sql(col, id, main_tab)}) AS a LEFT JOIN {cw_tab} AS b ON a.token = b.token) 
                GROUP BY idnum) AS b 
            ON a.{id}=b.idnum)
    """
    return str_out


# returns sql string for cleaned (only alphanumeric chars) institution name given column name of institution
def inst_clean_regex_sql(col):
    str_out = f"""
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    -- strip accents and convert to lowercase
                    strip_accents(lower({col})), 
                -- remove anything in parantheses
                '\\s*(\\(|\\[)[^\\)\\]]*(\\)|\\])\\s*', ' ', 'g'), 
            -- remove apostrophes, periods
            $$'|’|\\.$$, '', 'g'),
        -- convert 'and' symbols to text
        '\\s?(&|\\+)\\s?', ' and ', 'g'), 
    -- remove any other punctuation and replace w space
    '[^A-z0-9\\s]', ' ', 'g')
    """

    return f"TRIM(REGEXP_REPLACE({str_out}, '\\s+', ' ', 'g'))" 

# returns sql string for cleaned major/field of study given column name
def field_clean_regex_sql(col):
    str_out = f"""
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    -- strip accents and convert to lowercase
                    strip_accents(lower({col})), 
            -- remove apostrophes, periods
            $$'|’|\\.$$, '', 'g'),
        -- convert 'and' symbols to text
        '\\s?(&|\\+)\\s?', ' and ', 'g'), 
    -- remove any other punctuation and replace w space
    '[^A-z0-9/\\s]', ' ', 'g'),
    '(^|\\s)(engrg|engr|engineeri)($|\\s)', ' engineering ', 'g'),
    '(^|\\s)(compu|comp|compute)($|\\s)', ' computer ', 'g'),
    '(^|\\s)(elec|elecs)($|\\s)', ' electrical ', 'g'),
    '(^|\\s)(sci)($|\\s)', ' science ', 'g'),
    '(^|\\s)(info)($|\\s)', ' information ', 'g'),
    '(^|\\s)(sys|syss)($|\\s)', ' systems ', 'g'),
    '(^|\\s)(tec|techno|tech)($|\\s)', ' technology ', 'g'),
    '(^|\\s)(mech)($|\\s)', ' mechanical ', 'g'),
    '(^|\\s)(m\\s?a|m\\s?s|b\\s?s|b\\s?a|[0-9]+|m\\s?d|p\\s?h\\s?d|bachelor|bachelors|master|masters|in|and|j\\s?d|degree)($|\\s)', ' ', 'g')
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
        '\\s?(&|\\+)\\s?', ' and ', 'g'), 
    -- remove any other punctuation and replace w space
    '[^A-z0-9\\s]', ' ', 'g')
    """
    return f"TRIM(REGEXP_REPLACE({str_out}, '\\s+', ' ', 'g'))" 

# returns sql string for cleaned degree type
# (first look for high school, then if revelio has cleaned degree field, use that. otherwise look for keywords for non-degree program, then keywords/symbols for bachelors, then masters, then associate, then call anything starting with B bachelor, M master, (as long as followed by other capitals) else NA)
def degree_clean_regex_sql():
    str_out = f"""
        CASE 
            WHEN lower(university_raw) ~ '.*(high\\s?school).*' OR (degree = '<NA>' AND university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?)$') THEN 'High School' 
            WHEN degree != '<NA>' THEN degree
            WHEN lower(degree_raw) ~ '.*(cert|credential|course|semester|exchange|abroad|summer|internship|edx|cdl|coursera|udemy).*' OR lower(university_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera|udemy).*' OR lower(field_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera|udemy).*' THEN 'Non-Degree'
            WHEN (lower(degree_raw) ~ '.*(undergrad).*') OR (degree_raw ~ '.*(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?|B\\.?I\\.?S\\.?).*') OR degree_raw ~ '^B\\.?\\s?S\\.?.*' OR lower(field_raw) ~ '.*bachelor.*' OR lower(degree_raw) ~ '.*bachelor.*' THEN 'Bachelor'
            WHEN lower(degree_raw) ~ '.*(master).*' OR degree_raw ~ '^M\\.?(Eng|Sc|A)\\.?.*' THEN 'Master'
            WHEN degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?|M\\.?Eng|M\\.?Sc|M\\.?A).*' OR lower(field_raw) ~ '.*master.*' OR lower(degree_raw) ~ '.*master.*' THEN 'Master'
            WHEN lower(field_raw) ~ '.*(associate).*' OR degree_raw ~ 'A\\.?\\s?A\\.?.*' THEN 'Associate' 
            WHEN degree_raw ~ '^B\\.?\\s?[A-Z].*' THEN 'Bachelor'
            WHEN degree_raw ~ '^M\\.?\\s?[A-Z].*' THEN 'Master'
            WHEN degree IS NULL THEN 'Missing'
            ELSE degree END
    """
    return str_out

def degree_clean_degree_missing_regex_sql():
    str_out = f"""
        CASE 
            WHEN lower(university_raw) ~ '.*(high\\s?school).*' OR lower(degree_raw) ~ '.*(high\\s?school).*' OR lower(field_raw) ~ '.*(high\\s?school).*' OR lower(degree_raw) ~ 'g\\.?e\\.?d\\.?' THEN 'High School' 
            WHEN lower(degree_raw) ~ '.*(cert|credential|course|semester|exchange|abroad|summer|internship|edx|cdl|coursera|udemy).*' OR lower(university_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera|udemy).*' OR lower(field_raw) ~ '.*(edx|course|credential|semester|exchange|abroad|summer|internship|certificat|coursera|udemy).*' THEN 'Non-Degree'
            WHEN (lower(degree_raw) ~ '.*(undergrad).*') OR (degree_raw ~ '.*\\b(B\\.?A\\.?|B\\.?S\\.?C\\.?E\\.?|B\\.?Sc\\.?|B\\.?A\\.?E\\.?|B\\.?Eng\\.?|A\\.?B\\.?|S\\.?B\\.?|B\\.?B\\.?M\\.?|B\\.?I\\.?S\\.?)\\b.*') OR degree_raw ~ '^B\\.?\\s?S\\.?.*' OR lower(field_raw) ~ '.*bachelor.*' OR lower(degree_raw) ~ '.*(bachelor|baccalauréat).*' THEN 'Bachelor'
            WHEN lower(degree_raw) ~ '.*(master).*' OR degree_raw ~ '^M\\.?(Eng|Sc|A)\\.?.*' THEN 'Master'
            WHEN degree_raw ~ '.*(M\\.?S\\.?C\\.?E\\.?|M\\.?P\\.?A\\.?|M\\.?Eng|M\\.?Sc|M\\.?A).*' OR lower(field_raw) ~ '.*master.*' OR lower(degree_raw) ~ '.*master.*' THEN 'Master'
            WHEN lower(field_raw) ~ '.*(associate).*' OR degree_raw ~ 'A\\.?\\s?A\\.?.*' OR lower(degree_raw) ~ '.*associate.*' OR lower(field_raw) ~ '.*associate.*' THEN 'Associate' 
            WHEN lower(degree_raw) ~ '.*(doctor|ph\\.?\\s?d\\.?|d\\.?o\\.?|dvm|jd).*' OR degree_raw ~ '.*(Ph\\.?D\\.?|D\\.?O\\.?|J\\.?D\\.?).*' THEN 'Doctor'
            WHEN (university_raw ~ '.*(HS| High| HIGH| high|H\\.S\\.|S\\.?S\\.?C|H\\.?S\\.?C\\.?|I\\.?B\\.?)$') THEN 'High School' 
            WHEN degree_raw ~ '^B\\.?\\s?[A-Z].*' THEN 'Bachelor'
            WHEN degree_raw ~ '^M\\.?\\s?[A-Z].*' THEN 'Master'
            ELSE 'Missing' END
    """
    return str_out

# indicator for whether major is STEM or not
# logic: NOT if hs/nondegree, JD/MD?
#       YES if engineering
#       UNSURE if architecture, business, education, finance, medicine, accounting
#       cases to exclude: criminal justice, communication, human resources,
#           advertising, theater/television, history, design, government, humanities, journalism, english, administration, general? 
#       cases to include: information ? , pharmacy, psychology, geography, math, chemistry, biology, enivronment
def stem_ind_regex_sql():
    str_out = f"""
        CASE 
            WHEN field IN ('Engineering', 'Biology', 'Statistics', 'Chemistry', 'Medicine', 'Mathematics', 'Physics', 'Information Technology') THEN 1
            WHEN lower(field_raw) ~ '.*(engineer|information|math|chem|bio|pharm|psych|geography|environment|animation|cyber|comput|intelligence|data|aero|technol|software|web.*dev|energy|astronomy|robotics|hardware|machine learning|deep learning|nuclear|materials|earth|bsee|msee|game.*design|electr).*' THEN 1
            WHEN lower(degree_raw) ~ '.*(engineer|information|math|chem|bio|pharm|psych|geography|environment|animation|cyber|comput|intelligence|data|aero|technol|software|web.*dev|energy|astronomy|robotics|hardware|machine learning|deep learning|nuclear|materials|earth|bsee|msee|game.*design|electr).*' THEN 1
            WHEN lower(field_raw) ~ '.*science.*' AND NOT (lower(field_raw) ~ '.*(political|social|exercise|arts and|nursing|police) science.*') THEN 1
            WHEN field IN ('Nursing', 'Marketing', 'Law', 'Economics', 'Finance', 'Accounting', 'Business', 'Education', 'Architecture') THEN 0
            WHEN lower(field_raw) ~ '.*(studies|design|government|humanities|journalism|english|language|administration|history|advertising|education|marketing|nursing|law|economics|finance|accounting|architecture|linguistics|communication|theater|theatre|fashion|philosophy|politic|international|public|global|photography|sociology|religion|theology|dance|music|drama|writing|therapy|linguistics|media|administration|art|exercise|business|commerce|film|liberal|esthetic|anthropology|criminal|social|sales|health).*' THEN 0
            ELSE NULL
        END
    """
    return str_out 

# returns sql string for cleaned full name: removes everything after comma, removes anything 2-4 characters all caps as long as entire name not all caps (get rid of degrees or titles) [TODO: currently missing things like MSEd], then removes anything in parentheses at end, removes PhD specifically, removes plus specifically, then converts to title case (NOTE: title case function must be read into con)
def fullname_clean_regex_sql(col):
    str_out = f"""
        REGEXP_REPLACE(
        REGEXP_REPLACE(
        title(TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                    CASE WHEN {col} ~ '.*[a-z].*' THEN 
                        REGEXP_REPLACE(
                            REGEXP_REPLACE({col}, ',.*$', '', 'g'), 
                            '\\s([A-Z]\\.?){{2,4}}$', '', 'g')
                        ELSE REGEXP_REPLACE({col}, ',.*$', '', 'g') END, 
                    '\\s?\\(.*\\)$', '', 'g'), 
                'P\\.?h\\.?D\\.?', '', 'g'), 
            ' +', ' ', 'g'), 
        )),
        '\\s+[A-Z]\\.?\\s?$', '', 'g'),
        '\\s+[A-Z]\\.?\\s+', ' ', 'g')
    """
    return str_out 

# DEPRECATED: cleaning output from nanat (returns sql code for new column where each row is unique name x [nat, prob])
# def nanats_to_long(col):
#     str_out = f"""REGEXP_SPLIT_TO_ARRAY(UNNEST(REGEXP_SPLIT_TO_ARRAY(REGEXP_REPLACE(REGEXP_REPLACE({col}, '(\\[\\[|\\]\\])', '', 'g'), '",?', '', 'g'), '\\], \\[')), ' ')"""

#     return str_out

def get_est_yob():
    str_out = """
    CASE 
        -- if high school available
        WHEN (MAX(
            CASE 
                WHEN degree_clean = 'High School' 
                    THEN 1 ELSE 0 END
            ) OVER(PARTITION BY user_id)) = 1 
            THEN MAX(
                CASE 
                    WHEN degree_clean = 'High School' 
                        THEN SUBSTRING(ed_enddate, 1, 4)::INT - 18 
                    ELSE NULL 
                    END
                ) OVER(PARTITION BY user_id) 
        -- otherwise take bach/associate/missing 
        ELSE MIN(
            CASE 
                WHEN degree_clean = 'Non-Degree' OR degree_clean = 'Master' OR degree_clean = 'Doctor' OR degree_clean = 'MBA' 
                    THEN NULL 
                WHEN ed_startdate IS NOT NULL 
                    THEN SUBSTRING(ed_startdate, 1, 4)::INT - 18 
                WHEN ed_enddate IS NOT NULL AND NOT degree_clean = 'Associate' 
                    THEN SUBSTRING(ed_enddate, 1, 4)::INT - 23 
                WHEN ed_enddate IS NOT NULL 
                    THEN SUBSTRING(ed_enddate, 1, 4)::INT - 21 
                END
            ) OVER(PARTITION BY user_id) 
        END"""
    return str_out

# given table with list of some position/education type thing with start and enddate, returns sql string long on year x position for year t in (t_ref + t0, t_ref + t1) if t falls between startyr and endyr EXCLUDE({','.join([f"x.{joinid.strip()}, time.{joinid.strip()}" for joinid in joinids.split(',')])}),  
def long_by_year(tab, t0, t1, t_ref, enddatenull, startdatecol = 'startdate', enddatecol = 'enddate', joinids = 'user_id'):

    str_out = f"""
        SELECT {', '.join([f"CASE WHEN time.{joinid.strip()} IS NULL THEN x.{joinid.strip()} ELSE time.{joinid.strip()} END AS {joinid.strip()}" for joinid in joinids.split(',')])}, *,
        -- variable for fraction of year covered by position
            (LEAST(({t_ref} + time.t || '-12-31')::DATE, {enddatecol}::DATE) - GREATEST(({t_ref} + time.t || '-01-01')::DATE, {startdatecol}::DATE) + 1)/(({t_ref} + time.t || '-12-31')::DATE - ({t_ref} + time.t || '-01-01')::DATE + 1)
            AS frac_t
        FROM (SELECT generate_series AS t, {joinids} FROM generate_series({t0}, {t1}) CROSS JOIN (SELECT {joinids} FROM {tab} GROUP BY {joinids})) AS time
        FULL OUTER JOIN
        (SELECT *,
            SUBSTRING({startdatecol}, 1, 4)::INT AS startyr, 
            CASE WHEN {enddatecol} IS NULL THEN {enddatenull} ELSE SUBSTRING({enddatecol}, 1, 4)::INT END AS endyr FROM {tab}
        ) AS x 
        ON {t_ref} + time.t BETWEEN x.startyr AND x.endyr AND {' AND '.join([f"time.{joinid.strip()} = x.{joinid.strip()}" for joinid in joinids.split(',')])}
    """
    return str_out
    
## PYTHON FUNCTIONS
# randomly sample groups
def sample_groups(df, groupidcol, n):
    return df.loc[df[groupidcol].isin(np.random.permutation(df[groupidcol].unique())[:n])]

# helper function to get standardized country name
def get_std_country(country, dict):
    if country is None:
        return None 
    
    if country in dict.keys():
        return dict[country]
    
    if country in dict.values():
        return country 
    
    return "Invalid Country"

# cleaning country from gmaps json
def get_gmaps_country(adr, dict):
    if adr is None:
        return None 
    
    #print(adr)
    stub = re.search(', ([A-z\\s]+)[^,]*$', adr)
    if stub is not None:
        if stub.group(1) in dict.keys():
            return dict[stub.group(1)]
        elif stub.group(1) in dict.values():
            return stub.group(1)

    country_search = '(' + '|'.join(dict.values()) + ')'
    country_match = re.search(country_search, adr)

    if country_match is not None:
        #print(f"indirect match: {country_match.group(1)}")
        return country_match.group(1)
    
    return "No valid country match found"

def get_country_subregion(country, countrycw_dict, regioncw_dict):
    country_clean = get_std_country(country, countrycw_dict)

    if country_clean is None or country_clean == "Invalid Country":
        return "Invalid Country"
    
    if country in regioncw_dict.keys():
        return regioncw_dict[country]
    
    return 'Invalid Country'


# cleaning name2nat output
def get_all_nanats(str):
    if str is None:
        return []
    items = re.sub('\\\\u00e9','e', re.sub('(\\[\\[|\\]\\])','',str)).split('], [')
    out = []
    for s in items:
        if re.search('^"([A-z\\s\\-]+)", ', s) is None or re.search('", ([0-9\\.e\\-]+)$', s) is None:
            print(s)
        else:
            nat = re.search('^"([A-z\\s\\-]+)", ', s).group(1)
            prob = float(re.search('", ([0-9\\.e\\-]+)$', s).group(1))
            out = out + [get_std_country(nat)]
    return out
    # return [[dict[re.search('^"([A-z]+)", ', s).group(1)], float(re.search('", ([0-9\\.e\\-]+)$', s).group(1))] for s in items]

def get_all_nanat_probs(str):
    if str is None:
        return []
    items = re.sub('\\\\u00e9','e', re.sub('(\\[\\[|\\]\\])','',str)).split('], [')
    out = []
    for s in items:
        if re.search('^"([A-z\\s\\-]+)", ', s) is None or re.search('", ([0-9\\.e\\-]+)$', s) is None:
            print(s)
        else:
            nat = re.search('^"([A-z\\s\\-]+)", ', s).group(1)
            prob = float(re.search('", ([0-9\\.e\\-]+)$', s).group(1))
            out = out + [prob]
    return out


# breaking up large data tasks and saving intermediate output
# function to merge chunks
def chunk_merge(filestub, j, outfile = "", verbose = False):
    t0 = time.time()
    merged = []
    if verbose:
        print("Retrieving and merging chunks...")

    for i in range(j):
        chunk = pd.read_parquet(f"{filestub}{i}.parquet")
        merged = merged + [chunk]
    
    merged_all = pd.concat(merged)

    if outfile != "":
        if verbose:
            print(f"Saving to {outfile}...")
        merged_all.to_parquet(outfile)
    
    t1 = time.time()
    if verbose:
        print(f"Done! Time Elapsed: {round((t1-t0), 2)} s")

    return merged_all

# function to recursively iterate over j chunks of df until chunks are of desired size

# function to iterate over j chunks of userids (splitting each chunk further into k chunks)
def chunk_query(df, j, fun, jstart = 0, outpath = None, d = 10000, verbose = False, extraverbose = False):
    # ---- initialize / propagate progress bars ----
    # We detect "root" call by lack of private pbars in locals()
    _pb_rows   = getattr(chunk_query, "_pb_rows", None)
    _root_owner = False

    #total number of items
    n = df.shape[0] 

    # if no observations, return None
    if n == 0:
        return(None)
    
    # Create bars once at the root
    if _pb_rows is None:
        from tqdm.auto import tqdm 
        n_tot = n
        if jstart > 0:
            n_tot = n_tot - (jstart * int(np.ceil(n / j)))
        chunk_query._pb_rows   = tqdm(total=n_tot,             desc="Rows processed", unit="row")
        _pb_rows   = chunk_query._pb_rows
        _root_owner = True  # we own these bars and must close them

    # base case: if df has nrow <= d and not first call then perform function on df and return
    if n <= d and outpath is None:
        out = fun(df)
        # progress updates
        if _pb_rows is not None:
            _pb_rows.update(n)
        
        # close if we created bars at root
        if _root_owner:
            _pb_rows.close()
            chunk_query._pb_rows   = None

        return out
    
    # Otherwise, split into j chunks and recurse
    k = int(np.ceil(n / j))
    if verbose:
        print("\n----------------------------------------")
        print(f"Iterating over {j} chunks of {n} items ", end='')
        if outpath is None:
            print("and not saving intermediate output")
        else:
            print(f"and saving intermediate output to {outpath}")
        print("----------------------------------------")

    # if not saving output, recursively run function on chunks and concatenate into df
    if outpath is None:
        chunks = []
        for i in range(jstart, j):
            if verbose:
                print(f"\nchunk #{i+1} of {j}", end = '')
            t0 = time.time()

            start = k * i
            stop  = k * (i + 1)
            if start >= n:
                break

            subdf = df.iloc[start:stop]
            newchunk = chunk_query(subdf, j = j, fun = fun, outpath = None, d = d, verbose = extraverbose)
            
            if isinstance(newchunk, pd.DataFrame) and not newchunk.empty:
                newchunk = newchunk.loc[:, newchunk.notna().any(axis=0)]
                if newchunk.shape[0] != 0 and newchunk.shape[1] != 0:
                    chunks = chunks + [newchunk]
            t1 = time.time()

            if verbose:
                print(f"completed in {round((t1-t0)/60, 2)} min", end = '')

        # close bars if root
        if _root_owner and _pb_rows is not None:
            _pb_rows.close()
            chunk_query._pb_rows   = None

        if len(chunks)>0:
            return pd.concat(chunks)
        else:
            return None
        
    # if saving output, just recursively run query
    else: 
        for i in range(jstart, j):
            if verbose:
                print(f"\nchunk #{i+1} of {j}", end = '')
            t0 = time.time()
            start = k * i
            stop  = k * (i + 1)
            if start >= n:
                break

            subdf = df.iloc[start:stop]
            newchunk = chunk_query(subdf, j = j, fun = fun, outpath = None, d = d, verbose = extraverbose)

            if isinstance(newchunk, pd.DataFrame) and not newchunk.empty:
                newchunk.to_parquet(f"{outpath}{i}.parquet")
            
            else:
                print("DF empty, not saved")

            t1 = time.time() 

            if verbose:
                print(f"completed in {round((t1-t0)/60, 2)} min")
        
        # close bars if root
        if _root_owner and _pb_rows is not None:
            _pb_rows.close()
            chunk_query._pb_rows   = None

        return None
    
## FUZZY MERGE

def _norm(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_join_lev(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    *,
    threshold: float = 0.88,     # similarity in [0,1]
    top_n: int = 1,              # keep best n matches per left row
    block_key: str | None = None,# optional column in both to reduce comparisons
    normalize: bool = True,
    keep_score: bool = True,
) -> pd.DataFrame:
    """
    Fuzzy join using Levenshtein normalized similarity (RapidFuzz).
    Returns one row per kept match (duplicates left rows when top_n > 1).
    """
    L = left.copy()
    R = right.copy()

    # normalize keys
    lkey, rkey = left_on, right_on
    if normalize:
        L[lkey + "_n"] = L[left_on].map(_norm)
        R[rkey + "_n"] = R[right_on].map(_norm)
        lkey, rkey = lkey + "_n", rkey + "_n"

    # optional blocking
    if block_key is not None:
        if block_key not in L.columns or block_key not in R.columns:
            raise ValueError(f"block_key '{block_key}' must exist in both dataframes")
        groups = np.intersect1d(L[block_key].dropna().unique(), R[block_key].dropna().unique())
        blocks = ((L[L[block_key] == g], R[R[block_key] == g]) for g in groups)
    else:
        blocks = ((L, R),)

    out_parts = []

    for Lp, Rp in blocks:
        if Lp.empty or Rp.empty:
            continue

        left_vals  = Lp[lkey].fillna("").tolist()
        right_vals = Rp[rkey].fillna("").tolist()

        # Levenshtein normalized similarity in [0,1] (do NOT divide by 100)
        sim = process.cdist(
            left_vals, right_vals,
            scorer=distance.Levenshtein.normalized_similarity,
            workers=-1
        )

        if top_n == 1:
            # best match per left row
            jmax = sim.argmax(axis=1)                      # shape (m_left,)
            best = sim[np.arange(sim.shape[0]), jmax]      # shape (m_left,)
            keep = best >= threshold

            if keep.any():
                left_sel  = Lp.loc[keep].reset_index(drop=True)
                right_sel = Rp.iloc[jmax[keep]].reset_index(drop=True)

                merged = pd.concat(
                    [left_sel.add_suffix("_left"), right_sel.add_suffix("_right")],
                    axis=1
                )
                if keep_score:
                    merged["_lev_sim"] = best[keep]
                out_parts.append(merged)

        else:
            # top-n candidates per left row
            k = min(top_n, sim.shape[1])
            # indices of top-k columns (unsorted order OK; we'll mask anyway)
            top_idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
            rows, cols = np.indices(top_idx.shape)
            top_scores = sim[rows, top_idx]                # shape (m_left, k)

            keep_mask = top_scores >= threshold
            if keep_mask.any():
                # gather matched row indices
                li = Lp.index.values[rows[keep_mask]]
                ri = Rp.index.values[top_idx[keep_mask]]

                left_multi  = Lp.loc[li].reset_index(drop=True)
                right_multi = Rp.loc[ri].reset_index(drop=True)

                matched = pd.concat(
                    [left_multi.add_suffix("_left"), right_multi.add_suffix("_right")],
                    axis=1
                )
                if keep_score:
                    matched["_lev_sim"] = top_scores[keep_mask]
                out_parts.append(matched)

    if not out_parts:
        cols = list(left.add_suffix("_left").columns) + list(right.add_suffix("_right").columns)
        if keep_score:
            cols += ["_lev_sim"]
        return pd.DataFrame(columns=cols)

    return pd.concat(out_parts, ignore_index=True)

# Function that weights levenshtein and jaro-winkler metrics
def fuzzy_join_lev_jw(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    *,
    weights=(0.6, 0.4),            # (lev_weight, jw_weight)
    threshold: float = 0.88,       # combined cutoff in [0,1]
    top_n: int = 1,                # keep top-n matches per left row
    jw_top_k: int = 25,            # compute JW for top-k Levenshtein candidates
    block_key: str | None = None,  # optional column in both to reduce comparisons
    normalize: bool = True,
    keep_subscores: bool = True,
) -> pd.DataFrame:

    lev_w, jw_w = float(weights[0]), float(weights[1])
    if lev_w < 0 or jw_w < 0:
        raise ValueError("weights must be non-negative")
    if lev_w + jw_w == 0:
        raise ValueError("weights sum to zero")
    s = lev_w + jw_w
    lev_w, jw_w = lev_w / s, jw_w / s

    L = left.copy()
    R = right.copy()

    # normalize keys
    lkey, rkey = left_on, right_on
    if normalize:
        L[lkey + "_n"] = L[left_on].map(_norm)
        R[rkey + "_n"] = R[right_on].map(_norm)
        lkey, rkey = lkey + "_n", rkey + "_n"

    # optional blocking
    if block_key is not None:
        if block_key not in L.columns or block_key not in R.columns:
            raise ValueError(f"block_key '{block_key}' must exist in both dataframes")
        groups = np.intersect1d(L[block_key].dropna().unique(), R[block_key].dropna().unique())
        blocks = ((L[L[block_key] == g], R[R[block_key] == g]) for g in groups)
    else:
        blocks = ((L, R),)

    out_parts = []

    for Lp, Rp in blocks:
        if Lp.empty or Rp.empty:
            continue

        lv = Lp[lkey].fillna("").tolist()
        rv = Rp[rkey].fillna("").tolist()

        # Levenshtein normalized similarity in [0,1]
        lev_sim = process.cdist(
            lv, rv,
            scorer=distance.Levenshtein.normalized_similarity,
            workers=-1
        )
        m_left, m_right = lev_sim.shape
        if m_left == 0 or m_right == 0:
            continue

        k = min(jw_top_k, m_right)

        if top_n == 1:
            # top-k candidates by Levenshtein per row
            top_idx = np.argpartition(-lev_sim, kth=k-1, axis=1)[:, :k]

            # compute JW only for those candidates
            jw_sim_top = np.zeros((m_left, k), dtype=float)
            for i in range(m_left):
                a = lv[i]
                candidates = [rv[j] for j in top_idx[i]]
                jw_sim_top[i, :] = [jellyfish.jaro_winkler_similarity(a, b) for b in candidates]

            # combine scores on the candidate set
            combo_top = lev_w * lev_sim[np.arange(m_left)[:, None], top_idx] + jw_w * jw_sim_top

            best_pos_in_k = combo_top.argmax(axis=1)
            best_cols     = top_idx[np.arange(m_left), best_pos_in_k]
            best_scores   = combo_top[np.arange(m_left), best_pos_in_k]

            keep = best_scores >= threshold
            if keep.any():
                left_sel  = Lp.loc[keep].reset_index(drop=True)
                right_sel = Rp.iloc[best_cols[keep]].reset_index(drop=True)

                merged = pd.concat(
                    [left_sel.add_suffix("_left"), right_sel.add_suffix("_right")],
                    axis=1
                )
                merged["_score"] = best_scores[keep]

                if keep_subscores:
                    # recover component scores for kept pairs
                    lev_keep = lev_sim[keep, :][np.arange(keep.sum()), best_cols[keep]]
                    # No walrus inside comprehension: precompute arrays, then list-comp
                    lv_np = np.array(lv, dtype=object)[keep]
                    rv_np = Rp[rkey].iloc[best_cols[keep]].astype(str).tolist()
                    jw_keep = [jellyfish.jaro_winkler_similarity(a, b) for a, b in zip(lv_np, rv_np)]
                    merged["_lev_sim"] = lev_keep
                    merged["_jw_sim"]  = jw_keep

                out_parts.append(merged)

        else:
            # keep up to top_n candidates per left row above threshold
            k_for_keep = min(max(k, top_n), m_right)
            top_idx = np.argpartition(-lev_sim, kth=k_for_keep-1, axis=1)[:, :k_for_keep]
            rows, cols = np.indices(top_idx.shape)
            lev_top = lev_sim[rows, top_idx]

            jw_top = np.zeros_like(lev_top, dtype=float)
            for i in range(m_left):
                a = lv[i]
                candidates = [rv[j] for j in top_idx[i]]
                jw_top[i, :] = [jellyfish.jaro_winkler_similarity(a, b) for b in candidates]

            combo = lev_w * lev_top + jw_w * jw_top
            keep_mask = combo >= threshold
            if keep_mask.any():
                li = Lp.index.values[rows[keep_mask]]
                ri = Rp.index.values[top_idx[keep_mask]]

                left_multi  = Lp.loc[li].reset_index(drop=True)
                right_multi = Rp.loc[ri].reset_index(drop=True)

                matched = pd.concat(
                    [left_multi.add_suffix("_left"), right_multi.add_suffix("_right")],
                    axis=1
                )
                matched["_score"] = combo[keep_mask]
                if keep_subscores:
                    matched["_lev_sim"] = lev_top[keep_mask]
                    matched["_jw_sim"]  = jw_top[keep_mask]
                out_parts.append(matched)

    if not out_parts:
        base_cols = list(left.add_suffix("_left").columns) + list(right.add_suffix("_right").columns)
        extra = ["_score"]
        if keep_subscores:
            extra += ["_lev_sim", "_jw_sim"]
        return pd.DataFrame(columns=base_cols + extra)

    return pd.concat(out_parts, ignore_index=True)


# ------------------------------------------------------------------
# CIP matching compatibility wrappers (implementation moved to 10_misc/cip_matching.py)


@lru_cache(maxsize=1)
def _load_cip_matching_module():
    module_path = Path(__file__).resolve().parent / "10_misc" / "cip_matching.py"
    spec = importlib.util.spec_from_file_location("cip_matching", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Unable to load CIP matching module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def match_fields_to_cip(*args: Any, **kwargs: Any):
    """Proxy to 10_misc.cip_matching.match_fields_to_cip."""
    module = _load_cip_matching_module()
    return module.match_fields_to_cip(*args, **kwargs)


def add_hardcoded_cip_match(*args: Any, **kwargs: Any):
    """Proxy to 10_misc.cip_matching.add_hardcoded_cip_match."""
    module = _load_cip_matching_module()
    return module.add_hardcoded_cip_match(*args, **kwargs)
