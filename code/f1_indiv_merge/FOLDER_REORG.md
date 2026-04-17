# FOLDER REORG: PLANNING

I want to reorganize this entire folder to be a self-contained pipeline (non-numbered, so separate from the main h1bworkers pipeline). 

## Ground Rules:
1) Scripts should be entirely self-contained, not calling external scripts in other folders (with the exception of external_* scripts or config or helpers in the parent code folder). 
2) Many of the scripts we're creating will be similar or identical to scripts in other folders (which I will reference where appropriate) -- copy code from these scripts, DO NOT call/import them.
3) As with other code in this repo, all scripts should (A) should use a yaml config file for all relevant tuneable parameters and (B) be runnable in an interactive iPython terminal.


## Pipeline structure:
### main folder
Rename f1_indiv_merge into a self-contained pipeline root. All folders below will be contained within this main folder. If necessary, create a separate src folder with source code within this main folder. 
- there should also be a progress.md file (see similar files in external repo revelio-cleaning) that tracks progress across the pipeline

### 01_f1_foia_clean
This folder should contain code that (1) takes as input a raw folder of F1 FOIA data, (2) combines all years into one dataset (see _read_foia_raw and _read_foia_all in company_shift_share/deps_foia_clean.py) and saves this raw combined file, (3) links people across years in the raw data and generates stable person_ids + corrects employment spells (as in f1_foia/foia_person_id_linkage.py) and saves this cleaned output.

### 02_rev_import -- HEAVY REWRITE, LEAVE AS SKELETON FOR NOW
This folder should contain code that imports raw user education and position data from LinkedIn. The overall structure will be similar to 02_revelio_indiv_clean/wrds_positions.py and wrds_users.py, but the filtering criteria will be very different. The existing files filter on rcid. The new files need to look at all users to determine who to import. 

- Preprocessing: Take the raw list of FOIA institution strings, clean and tokenize, and compile the core unique tokens (stripping common tokens) into one regex string (e.g. '(stanford|georgetown|mit|purdue|...)').
- Sharding: it will be extremely expensive to look through all users at once on a single pass. Instead, shard users by userid into deterministic groups of some specified (via parameter) size n. 
- For each group of users n: 
1. filter their individual_user_education to education objects that (A) either have university_country = US or university_country = None, (B) the degree is not High School or Associates (can be None), and (C) where the startdate is after 2000, the enddate is after 2004, or both dates are null. 
2. join the filtered education data to individual_user_education_raw, use regex + sql logic (see relevant functions in helpers.py) to further extract information on high school/non-degree educations and drop those. 
3. search the raw university names (and raw degree and field names, in case of erroneous entry into the wrong header) for the preprocessed regex string. 
4. take the final list of education objects that match to the preprocessed regex string, and return the unique list of user_ids.
- Final Import: after the userid list is constructed, run wrds_positions.py and wrds_users.py filtering on user_id rather than rcid. The returned columns should be the same (just for a different sample).

### 03_rev_crosswalks
This folder will first require the wrds_users file imported above to pass through the edu_object cleaning pipeline in the external repo revelio-cleaning. If this hasn't been done, warn the user and error out. Once it's been done, the cleaned mappings can be imported through external_us_school_matching.
This folder should generate four crosswalks: (1) f1-ipeds should map f1 institution rows to ipeds unitids. (2) rev-ipeds should map university_raw values to ipeds unitids. (3) field-cip should map revelio field_raw values to cip codes. (4) rcid-emp should map revelio rcids to f1 employers. 

### 04_rev_user_clean
This folder should follow similar logic as 02_revelio_indiv_clean/rev_users_clean.py. First, user names from the wrds pull should be passed through name2nat and nametrace. Then, this should be combined with the institution mappings above + position history to generate candidate match countries + associated country_scores for all users. The code should also apply the field_mapping and employer_mappings from the crosswalks above. The script should produce a user x ipeds school x degree x country of birth x field x employer dataset ???

### 05_indiv_merge
This folder should follow similar structure as 03_indiv_merge, but should use the logic in the current f1_indiv_merge.py file -- merging should be on school x degree x country of birth x field x employer, then raw merges should be aggregated up to the user_id x f1_person_id level (combining scores across employers), filtered on some basic scoring + timing of education spell, then scoring/ranking should rely mostly on employers, with support from other match variable quality.
