# File Description: Declaring global variables
# Author: Amy Kim
# Date Created: July 8 2025
import os 
import json

#######################
### DECLARING PATHS ###
#######################
# local
if os.environ.get('USER') == 'amykim':
    root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
    code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

    # separate in and out paths for reading/writing data
    wrds_in = f'{root}/data/wrds/wrds_in'
    wrds_out = f'{root}/data/wrds/wrds_out/jun26'

# malloy
elif os.environ.get('USER') == 'yk0581':
    root = "/home/yk0581"
    code = f"{root}/h1bworkers/code"

    # separate in and out paths for reading/writing data
    wrds_in = f'{root}/data/wrds/wrds_in'
    wrds_out = f'{root}/data/wrds/wrds_out/jun26'

# wrds
elif os.environ.get('USER') == 'amykimecon':
    root = "/home/princeton/amykimecon"
    code = "/home/princeton/amykimecon"

    # separate in and out paths for reading/writing data
    wrds_in = "/home/princeton/amykimecon/data"
    wrds_out = "/scratch/princeton/amykimecon"

# setting path
os.chdir(f"{code}")

# reading api keys as global
with open(f"{root}/data/private_configs/hud_api_key.json") as f:
    cfg = json.load(f)

with open(f"{root}/data/private_configs/openai_api_key.json") as f:
    openai_cfg = json.load(f)

HUD_API_KEY = cfg["api_key"]
OPENAI_API_KEY = openai_cfg["api_key"]

# import helper functions
import helpers as help

##################################
### IMPORTING HELPER FUNCTIONS ###
##################################
# import rev_indiv_clean_helpers as help

# con = ddb.connect()

# ## Creating DuckDB functions from python helpers
# #title case function
# con.create_function("title", lambda x: x.title(), ['VARCHAR'], 'VARCHAR')

# # country crosswalk function
# con.create_function("get_std_country", lambda x: help.get_std_country(x), ['VARCHAR'], 'VARCHAR')
