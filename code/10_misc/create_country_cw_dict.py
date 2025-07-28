# File Description: Creating country crosswalk dictionary
# Author: Amy Kim
# Date Created: Mon May 5 2025

# Imports and Paths
import pandas as pd
import json
import sys
import os
import re
from unidecode import unidecode

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

## HARMONIZING COUNTRIES
# reading in crosswalk of country codes
iso_cw = pd.read_csv(f"{root}/data/crosswalks/iso_country_codes.csv")

# cleaning full names to get 'standardized' country names
iso_cw['name_clean'] = iso_cw['name'].str.replace(',.*$','',regex =True).str.replace(' \\(.*\\)','',regex=True).apply(unidecode)

# initializing dict: appending full name, three-letter abbrev and two-letter abbrev, mapping to cleaned name
country_cw_dict = iso_cw[['name','name_clean']].set_index('name').T.to_dict('records')[0]|iso_cw[['alpha-3','name_clean']].set_index('alpha-3').T.to_dict('records')[0]|iso_cw[['alpha-2','name_clean']].set_index('alpha-2').T.to_dict('records')[0]

# manual corrections
nullkeys = []
for key, val in country_cw_dict.items():
    if val == "Brunei Darussalam":
        country_cw_dict[key] = 'Brunei'

    if val == 'Congo':
        country_cw_dict[key] = "Democratic Republic of the Congo"

    if val == "Lao People's Democratic Republic":
        country_cw_dict[key] = 'Laos'
    
    if val == "Puerto Rico":
        country_cw_dict[key] = 'United States'

    if val == "Russian Federation":
        country_cw_dict[key] = 'Russia'
    
    if val == "Syrian Arab Republic":
        country_cw_dict[key] = 'Syria'

    if val == "United Kingdom of Great Britain and Northern Ireland":
        country_cw_dict[key] = "United Kingdom"
    
    if val == "United States of America":
        country_cw_dict[key] = 'United States'
    
    if val == "Viet Nam":
        country_cw_dict[key] = 'Vietnam'

    if pd.isnull(key) == True:
        nullkeys = nullkeys + [key]

for key in nullkeys:
    country_cw_dict.pop(key)

country_cw_dict['DRC'] = 'Democratic Republic of the Congo'
country_cw_dict['NA'] = 'Namibia'
country_cw_dict['Czech Republic'] = 'Czechia'
country_cw_dict["Korea, Democratic People's Republic of"] = "North Korea"
country_cw_dict['PRK'] = "North Korea"
country_cw_dict['KP'] = "North Korea"
country_cw_dict["Korea, Republic of"] = "South Korea"
country_cw_dict['KOR'] = "South Korea"
country_cw_dict['KR'] = "South Korea"
country_cw_dict['Virgin Islands (British)'] = 'British Virgin Islands'
country_cw_dict['VGB'] = 'British Virgin Islands'
country_cw_dict['VIR'] = 'U.S. Virgin Islands'
country_cw_dict['Virgin Islands (U.S.)'] = 'U.S. Virgin Islands'
country_cw_dict['East Timor'] = 'Timor-Leste'
country_cw_dict['UAE'] = "United Arab Emirates"
country_cw_dict['UK'] = "United Kingdom"
country_cw_dict['USA'] = 'United States'
country_cw_dict['Kashmir'] = "India"
country_cw_dict['Burkinabe'] = "Burkina Faso"
country_cw_dict['Tibetan'] = 'Nepal'
country_cw_dict['Tamil'] = 'Sri Lanka'
country_cw_dict['American'] = 'United States'
country_cw_dict['French'] = 'France'
country_cw_dict['XK'] = 'Kosovo'
country_cw_dict['XKX'] = 'Kosovo'
country_cw_dict['XKK'] = 'Kosovo'
country_cw_dict['Kosovo'] = 'Kosovo'
country_cw_dict['Turkey'] = 'Turkiye'
country_cw_dict['Macedonia'] = 'North Macedonia'
country_cw_dict['Congo - Democratic Rep'] = 'Democratic Republic of the Congo'
country_cw_dict['Ivory Coast'] = "Cote d'Ivoire"
country_cw_dict['Swaziland'] = 'Eswatini'
country_cw_dict['Central African Republ'] = 'Central African Republic'
country_cw_dict['Vatican City'] = 'Italy'
country_cw_dict['Virgin Islands (UK)'] = 'British Virgin Islands'
country_cw_dict['Svalbard'] = 'Norway'
country_cw_dict['Congo - Republic of th'] = 'Democratic Republic of the Congo'
country_cw_dict['Burkinab\u00e9'] = 'Burkina Faso'
country_cw_dict['Basque'] = 'Spain'
country_cw_dict['Syriac'] = 'Syria'

# mapping nanat nationalities to country names
def get_nanat_country(nat, dict = country_cw_dict):
    print(f"NAT: {nat}")
    if nat in dict.keys():
        return dict[nat]
    
    if nat in dict.items():
        return nat
    
    if re.sub("n$","",nat) in dict.items():
        print(re.sub("n$","",nat))
        return re.sub("n$","",nat)
    
    stub = re.sub("(ian|ish|ese|e)$","",nat)
    print(stub)

    if len(stub) >= 3:
        for (key, value) in dict.items():
            if re.search(stub, key) is not None:
                valid = input(f"Is {key} the correct match? (Y/N) ")
                if valid == "Y":
                    return key
            if value != key and re.search(stub, value) is not None:
                valid = input(f"Is {value} the correct match? (Y/N) ")
                if valid == "Y":
                    return value
        
    return None 
            
unmatched = []
for nat in my_nanat.name2nats.keys():
    country = get_nanat_country(nat)
    if country is not None:
        country_cw_dict[nat] = country
    else:
        unmatched = unmatched + [nat]

for nat in unmatched:
    ctry = input(f"Nationality of {nat}: ")
    if ctry != "":
        country_cw_dict[nat] = ctry
        print(ctry)
    else:
        print("ERROR")


# EXISTING DICT (ONLY RUN IF UPDATING DICT, OTHERWISE COMMENT THIS PART OUT)
with open(f"{root}/data/crosswalks/country_dict.json", "r") as json_file:
    country_cw_dict_old = json.load(json_file)

    country_cw_dict = country_cw_dict|country_cw_dict_old

# export json 
with open(f"{root}/data/crosswalks/country_dict.json", "w") as json_file:
    json.dump(country_cw_dict, json_file)