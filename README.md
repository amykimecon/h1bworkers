# Project Overview

## Description and Inputs
Using H-1B visa data and Revelio LinkedIn data, merging to evaluate outcomes of H-1B lottery winners and losers.

## File Structure (updated 6/26/25)
1. Company Merge (code/01_employer_merge)
  - V1: R (currently using)
    - 1.0. Importing Revelio Employers (revelio_explore.R)
      - input: WRDS credentials
      - output: /data/int/revelio/companies_by_positions_locations.csv
    - 1.1. Deterministic Linking (employer_merge.R; merge_helper.R)
      - input: raw FOIA data; /data/int/revelio/companies_by_positions_locations.csv
      - output: good_match_ids_mar20.csv; dup_rcids_mar20.csv
        
  - V2: Python (not operational) -- **TODO: FIX**
    - 1.0. Importing Revelio Employers (revelio_import.py)
    - 1.1. Preprocessing Employer Data (employer_merge_preprocess.py; employer_merge_helpers.py)
    - 1.2. Merging Employer Data with Splink (employer_merge.py)
     
2. Revelio Individual Data Clean (code/02_revelio_indiv_clean)
  - 2.0.1. Importing Revelio User Info on WRDS Cloud (../batch_queries/revmerge_users.py; ../batch_queries/revmerge_users.sh)
      - input: good_match_ids_mar20.csv; dup_rcids_mar20.csv
      - output: rev_user_merge{j}.parquet for j in 1:10 [user x education info]
  - 2.0.1. Importing Revelio User Positions (get_rev_merged.py)
      - input: good_match_ids_mar20.csv; dup_rcids_mar20.csv
      - output: rev_merge_mar20.csv [user x position info]
  - 2.1.1. Institution Matching to Countries via OpenAlex, Geonames, Gmaps (clean_revelio_institutions.py; rev_indiv_clean_helpers.py)
      - input: rev_user_merge{j}.parquet for j in 1:10; raw openalex, geonames, gmaps data
      - output: /data/int/rev_inst_countries_jun25.parquet
  - 2.1.2. Getting Nationalities from Full Names using Name2Nat (rev_indiv_name2nat.py)
      - input: rev_user_merge{j}.parquet for j in 1:10; name2nat function
      - output: rev_names_withnat_jun26.parquet
  - 2.2. Final Clean (rev_users_clean.py)
      - input: rev_user_merge{j}.parquet for j in 1:10; rev_names_withnat; rev_inst_countries
      - output: rev_user_nats_final_apr15

3. Individual Merge (code/03_indiv_merge)
  - 3.1. Merging (indiv_merge_test.py)

4. Analysis

10. Misc Helpers (code/10_misc)
      - get_location_gmaps.py -- queries gmaps API for locations of institutions
      - create_country_cw_dict.py -- creates dictionary mapping alternate names of countries to standardized name
      - create_occ_cw.py -- creates crosswalks between revelio occupations and DOT occupation codes (used in H-1B)
      - name2nat_train.py -- trains model for guessing ethnicity from name (adapted from name2nat package)
   
