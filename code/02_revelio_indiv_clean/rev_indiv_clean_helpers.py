## HELPERS FOR CLEANING INDIVIDUAL REVELIO DATA
import json
import numpy as np
import re
import pandas as pd
from name2nat import Name2nat 
import os

# local
if os.environ.get('USER') == 'amykim':
    root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
    code = "/Users/amykim/Documents/GitHub/h1bworkers/code"
# malloy
elif os.environ.get('USER') == 'yk0581':
    root = "/home/yk0581"
    code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

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

# cleaning output from nanat (returns sql code for new column where each row is unique name x [nat, prob])
def nanats_to_long(col):
    str_out = f"""REGEXP_SPLIT_TO_ARRAY(UNNEST(REGEXP_SPLIT_TO_ARRAY(REGEXP_REPLACE(REGEXP_REPLACE({col}, '(\\[\\[|\\]\\])', '', 'g'), '",?', '', 'g'), '\\], \\[')), ' ')"""

    return str_out

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
    
## PYTHON FUNCTIONS
# randomly sample groups
def sample_groups(df, groupidcol, n):
    return df.loc[df[groupidcol].isin(np.random.permutation(df[groupidcol].unique())[:n])]

# Importing Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

# helper function to get standardized country name
def get_std_country(country, dict = country_cw_dict):
    if country is None:
        return None 
    
    if country in dict.keys():
        return dict[country]
    
    if country in dict.values():
        return country 
    
    return "Invalid Country"

# cleaning country from gmaps json
def get_gmaps_country(adr, dict = country_cw_dict):
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

# getting subregion from country
regioncw = pd.read_csv(f"{root}/data/crosswalks/iso_country_codes.csv")
regioncw['region_clean'] = np.where(pd.isnull(regioncw['intermediate-region']), regioncw['sub-region'], regioncw['intermediate-region'])
regioncw['country_clean'] = regioncw['name'].apply(get_std_country)
regioncw_dict = regioncw[['country_clean','region_clean']].set_index('country_clean').T.to_dict('records')[0]

def get_country_subregion(country, regioncw_dict = regioncw_dict):
    country_clean = get_std_country(country)

    if country_clean is None or country_clean == "Invalid Country":
        return "Invalid Country"
    
    if country_clean == 'Taiwan':
        return 'Eastern Asia'
    
    if country_clean == 'Kosovo':
        return 'Eastern Europe'
    
    if country in regioncw_dict.keys():
        return regioncw_dict[country]
    
    return 'Invalid Country'

# name2nat helper function
my_nanat = Name2nat()

def name2nat_fun(name, nanat = my_nanat):
    return json.dumps(nanat(name, top_n = 10)[0][1])

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