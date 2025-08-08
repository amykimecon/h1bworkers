# File Description: Creating crosswalk to get 
# Author: Amy Kim
# Date Created: Mon May 5 2025

# Imports and Paths
import pandas as pd
import json
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# Importing Country Codes Crosswalk
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict = json.load(json_file)

# getting subregion from country
regioncw = pd.read_csv(f"{root}/data/crosswalks/iso_country_codes.csv")
regioncw['region_clean'] = np.where(pd.isnull(regioncw['intermediate-region']), regioncw['sub-region'], regioncw['intermediate-region'])
regioncw['country_clean'] = regioncw['name'].apply(lambda x: help.get_std_country(x, country_cw_dict))
regioncw_dict = regioncw[['country_clean','region_clean']].set_index('country_clean').T.to_dict('records')[0]
regioncw_dict['Taiwan'] = 'Eastern Asia'
regioncw_dict['Kosovo'] = 'Eastern Europe'

# EXISTING DICT (ONLY RUN IF UPDATING DICT, OTHERWISE COMMENT THIS PART OUT)
# with open(f"{root}/data/crosswalks/subregion_dict.json", "r") as json_file:
#     regioncw_dict_old = json.load(json_file)

#     regioncw_dict = regioncw_dict|regioncw_dict_old

# export json 
with open(f"{root}/data/crosswalks/subregion_dict.json", "w") as json_file:
    json.dump(regioncw_dict, json_file)