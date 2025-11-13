import sys 
import os 
import pandas as pd
import openai 
import duckdb as ddb
import json
from typing import Iterable, Dict, Any, List
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(f"{os.path.dirname(os.path.dirname(__file__))}/10_misc")
from config import *
from cip_matching import match_fields_to_cip

con = ddb.connect()

# CONSTANTS
light_catalog = True 
samp_from_scratch = False
samp_size = 10
data_save_path = f"{root}/data/int/wrds_fields_for_cip_matching_nov2025_v2.csv"
response_save_path = f"{root}/data/int/wrds_cip_matching_llm_response_nov2025_v2.json"

###########################
## IMPORTING CIP CATALOG ##
###########################
# raw import
cip_catalog_df_raw = pd.read_csv(f"{root}/data/crosswalks/cip/CIPCode2020.csv", dtype=str)

# filtering to valid cip codes only (for this version, want only 6-digit CIP codes)
cip_catalog_df_filt = cip_catalog_df_raw[cip_catalog_df_raw["CIPCode"].str.match(re.compile(r'^="(\d{2}\.\d{4})"$'), na=False)].copy()

# building catalog
cip_catalog = [
    {
        "CIPCode": re.sub(r'^="(\d{2}\.\d{4})"$', r'\1', row["CIPCode"]),
        "CIPTitle": row["CIPTitle"],
        "CIPDefinition": row["CIPDefinition"],
        "Examples": row["Examples"],
    }
    for _, row in cip_catalog_df_filt.iterrows()
]

# building light catalog (way fewer tokens)
cip_catalog_light = [
    {
        "CIPCode": item["CIPCode"],
        "CIPTitle": item["CIPTitle"],
    }
    for item in cip_catalog
]

# setting catalog and catalog text based on light or full
catalog_text = "(code, title, definition, examples)"

if light_catalog:
    cip_catalog = cip_catalog_light
    catalog_text = "(code, title)"

##################################
## IMPORTING RAW EDUCATION INFO ##
##################################
if samp_from_scratch:
    # raw import of LinkedIn data using duckdb (very large)
    raw_fields = con.read_parquet(f"{root}/data/int/wrds_users_sep2.parquet")

    # cleaning/normalizing relevant columns
    clean_fields = con.sql(f"SELECT user_id, university_raw, field_raw, degree_raw, {help.field_clean_regex_sql('field_raw')} AS field_clean, {help.degree_clean_regex_sql()} AS degree_clean, ed_startdate, ed_enddate FROM raw_fields")

    # randomly sampling fields for matching
    fields_for_matching = con.sql("SELECT * FROM clean_fields WHERE field_raw IS NOT NULL AND (university_raw IS NOT NULL OR degree_raw IS NOT NULL) AND NOT degree_clean = 'High School'").df().sample(samp_size*500, random_state=1003)

    # performing initial cip matching (using external cip_matching module that uses deterministic rules)
    fields_matched = match_fields_to_cip(fields_for_matching, field_column = 'field_raw')

    # merging cip matches back to fields
    fields_for_llm = fields_for_matching.merge(fields_matched[['user_id', 'field_raw', 'cip_code', 'cip_title', 'cip_match_score', 'cip_match_source']], on = ['user_id', 'field_raw'], how = 'left')    

    # select five random rows without a successful cip match for testing
    test_samp_nomatch = fields_for_llm[(fields_for_llm['cip_match_source'] == "unmatched") & (fields_for_llm['field_clean'] != '')].sample(samp_size)

    # select five random rows with missing degrees for testing
    test_samp_missingdegree = fields_for_llm[(fields_for_llm['degree_clean'] == 'Missing') & (fields_for_llm['field_clean'] != '')].sample(samp_size)

    fields_for_llm['test_samp'] = 0
    fields_for_llm.loc[test_samp_nomatch.index, 'test_samp'] = 1
    fields_for_llm.loc[test_samp_missingdegree.index, 'test_samp'] = 1

    # saving file for reference
    fields_for_llm[fields_for_llm['test_samp'] == 1].to_csv(data_save_path, index = False)

else:
    # loading previously saved file
    fields_for_llm = pd.read_csv(data_save_path)

# test samp
test_samp = fields_for_llm[fields_for_llm['test_samp'] == 1].to_dict(orient="records")

##################################
## DECLARING PROMPT COMPONENTS ###
##################################
CATALOG_CONTEXT = json.dumps(cip_catalog, ensure_ascii=False)

SYSTEM_PROMPT = f"""You are an analyst mapping education records to U.S. CIP 2020 codes. 
You must choose from this catalog {catalog_text}:
{CATALOG_CONTEXT}
Return JSON per record with keys: user_id, cip_code, degree_type, confidence (0-1), reasoning.
"""

TEMPLATE = """user_id: {user_id}
  university_raw: {university_raw}
  degree_raw: {degree_raw}
  field_raw: {field_raw}
  ed_startdate: {ed_startdate}
  ed_enddate: {ed_enddate}
"""

## HELPER TO BUILD PROMPTS GIVEN DICT OF ROWS
def build_prompt(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    content = []
    for row in rows:
        content.append(
            TEMPLATE.format(
                user_id=row["user_id"],
                university_raw=row.get("university_raw", "N/A"),
                degree_raw=row.get("degree_raw", "N/A"),
                field_raw=row.get("field_raw", "N/A"),
                ed_startdate=row.get("ed_startdate", "N/A"),
                ed_enddate=row.get("ed_enddate", "N/A"),
            )
        )
    user_prompt = (
        "Select the best matching CIP code from the catalog above and the best matching degree type from High School, Bachelor, Associate, Master, Doctor, Non-Degree for each row. "
        "If there is insufficient information for either CIP or degree, set to null and explain why.\n\n"
        + "\n".join(content)
    )

    return user_prompt

# checking if prompt looks okay
print(build_prompt(test_samp))

###################
## QUERYING LLM ###
###################
client = openai.OpenAI(
    api_key=OPENAI_API_KEY
)

# saving system prompt to text
with open(f"{root}/data/int/blackfish_sample_system_prompt.txt", "w") as f:
    f.write(SYSTEM_PROMPT)
    
# saving queries to text
with open(f"{root}/data/int/blackfish_sample_input.txt", "w") as f:
    f.write(build_prompt(test_samp))

# safety guard before querying
proceed = input(f"About to query LLM for {len(test_samp)} records. Proceed? (y/n): ")
if proceed.lower() != 'y':
    print("Aborting LLM query.")
    sys.exit()
response = client.responses.create(
        model="gpt-5-mini",
        instructions = SYSTEM_PROMPT,
        input = build_prompt(test_samp),
        service_tier='flex'
    )

print(response.output_text)

# number of tokens used
print(f"Total tokens used: {response.usage.total_tokens}")

# write response to json
with open(response_save_path, "w") as f:
    json.dump(response.output_text, f, ensure_ascii=False, indent=4)

# read response 
response_df = pd.DataFrame(json.loads(response.output_text))

# merge response back to fields
fields_final = fields_for_llm.merge(response_df, on = 'user_id', how = 'left')