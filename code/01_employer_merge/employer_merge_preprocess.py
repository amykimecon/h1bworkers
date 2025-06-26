# File Description: Cleaning, Collapsing, Standardizing Employer Information from Revelio and H1B Data
# Author: Amy Kim
# Date Created: Wed Mar 5

# Imports and Paths
import duckdb as ddb
import time
import employer_merge_helpers as emh

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

#####################
# IMPORTING DATA
#####################
## Importing Revelio Data
rev_raw_file = con.read_parquet(f"{root}/data/int/revelio/revelio_agg.parquet")
con.sql("CREATE TABLE rev_raw AS SELECT * FROM rev_raw_file")

## Importing FOIA Data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")
con.sql("CREATE TABLE foia_raw AS SELECT * FROM foia_raw_file")

#################################
# CLEANING AND COLLAPSING DATA
#################################
## filtering revelio data
emh.create_replace_table(con=con, query="SELECT * FROM rev_raw WHERE n_positions_us > 0 AND (recent_startdate_global > '2018-01-01' OR recent_enddate_global > '2018-01-01')", table_out="rev_filt", show = False)

## filtering foia data
emh.create_replace_table(con=con, query="SELECT * FROM foia_raw WHERE NOT employer_name = '(b)(3) (b)(6) (b)(7)(c)'", table_out="foia_filt", show = False)

## collapsing Revelio data to company_name and rcid level, joining with rcid-level list of states/naics and counts
'''
structure of query:
    1. collapse all non-null locations by rcid into list with counts (rcid,      pos_locations as zipped list of {state, share of positions in state}, most common state)
    2. full join on rcid with table of collapsed non-null NAICS codes by rcid into list with counts (rcid, naics_codes as zipped list of {naics, share of positions labelled as naics}, most common naics code -- N.B. for revelio data, naics codes constant by rcid so will always be max one naics code per company)
    3a-d. full join on rcid with tables of collapsed 2-5 digit NAICS codes by rcid into list with counts (rcid, naics{num}_codes as zipped list of {naics, share of positions labelled as naics})
    4. right join on rcid with main list of companies: grouped by company name and rcid with n_positions_us and n_users_us
'''
rev_collapse_query = f"""
SELECT (ROW_NUMBER() OVER()) + 1000000 AS unique_id,
    locs.rcid as rawid, company, {emh.clean_company_string("company")} AS company_clean, n_users_us as n, 
    pos_locations as emp_locations, pos_locations as work_locations,  
    top_location as top_emp_location, top_location as top_work_location,
    naics_codes, naics2_codes, naics3_codes, naics4_codes, naics5_codes, top_naics_code, top_naics2_code, top_naics3_code, top_naics4_code, top_naics5_code
FROM (
    -- 1. collapse all locations by rcid into list with counts
    (SELECT rcid,
        list_transform(
            list_zip(
                array_agg(state ORDER BY n_positions DESC),
                array_agg(n_positions/n_positions_us ORDER BY n_positions DESC)
            ),
            x -> struct_pack(state := x[1], share := x[2])
        ) as pos_locations,
        array_agg(state ORDER BY n_positions DESC).list_extract(1) AS top_location
    FROM (SELECT state, rcid, sum(n_positions) AS n_positions, mean(n_positions_us) AS n_positions_us FROM rev_filt WHERE state IS NOT NULL GROUP BY state, rcid) 
    GROUP BY rcid
    ) AS locs
    FULL JOIN (
    -- 2. collapse all NAICS codes by rcid into list with counts
    {emh.group_by_naics_code_rev()}) AS naics
    ON locs.rcid = naics.rcid
    FULL JOIN (
    -- 3a. collapse all NAICS 2-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("2")}) AS naics2
    ON locs.rcid = naics2.rcid
    FULL JOIN (
    -- 3b. collapse all NAICS 3-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("3")}) AS naics3
    ON locs.rcid = naics3.rcid
    FULL JOIN (
    -- 3c. collapse all NAICS 4-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("4")}) AS naics4
    ON locs.rcid = naics4.rcid
    FULL JOIN (
    -- 3d. collapse all NAICS 5-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("5")}) AS naics5
    ON locs.rcid = naics5.rcid
    RIGHT JOIN (
    -- 4. main list of companies: by company name and rcid (n_positions_us and n_users_us should be constant across rcids)
    SELECT rcid, 
        company, MEAN(n_positions_us) AS n_positions_us , 
        MEAN(n_users_us) AS n_users_us
    FROM rev_filt
    GROUP BY rcid, company
    ) AS raw
    ON locs.rcid = raw.rcid
)
"""
emh.create_replace_table(con, rev_collapse_query, "rev_collapsed", show = False)

## collapsing FOIA data to company_name and FEIN level, joining with FEIN-level list of states/naics and counts
'''
structure of query:
    1. define temporary table e with counts of applications, wins, and substrings of NAICS codes by FEIN (not collapsed)
    2. collapse all non-null employer locations by FEIN into list with counts (FEIN, emp_locations as zipped list of {state, share of applications in state}, most common state)
    3. full join on FEIN with table of collapsed non-null worksite locations by FEIN into list with counts (FEIN, work_locations as zipped list of {state, share of wins with worksites in state}, most common state)
    4. full join on FEIN with table of collapsed non-null NAICS codes by FEIN into list with counts (FEIN, naics_codes as zipped list of {naics, share of positions labelled as naics}, most common naics code)
    4a-d. full join on FEIN with tables of collapsed 2-5 digit NAICS codes by FEIN into list with counts (FEIN, naics{num}_codes as zipped list of {naics, share of positions labelled as naics})
    5. right join on FEIN with main list of companies: grouped by (raw) company name and FEIN with number of apps, wins, and max reported number of us employees
'''
foia_collapse_query = f"""
-- 1. define employer_counts table with counts of applications, wins, and substrings of NAICS codes by FEIN
WITH employer_counts AS (
    SELECT FEIN, state_name, 
        COUNT(CASE WHEN NOT state_name = 'NA' THEN 1 END) OVER (PARTITION BY FEIN) AS n_apps_employer_tot, 
        worksite_state_name, 
        COUNT(CASE WHEN NOT worksite_state_name = 'NA' THEN 1 END) OVER (PARTITION BY FEIN) AS n_wins_worksite_tot, 
        NAICS_CODE, SUBSTR(NAICS_CODE, 1, 2) AS NAICS2_CODE, SUBSTR(NAICS_CODE, 1, 3) AS NAICS3_CODE, SUBSTR(NAICS_CODE, 1, 4) AS NAICS4_CODE, SUBSTR(NAICS_CODE, 1, 5) AS NAICS5_CODE, 
        COUNT(CASE WHEN NOT NAICS_CODE = 'NA' AND NOT NAICS_CODE = '999999' THEN 1 END) OVER (PARTITION BY FEIN) AS n_wins_naics_tot FROM foia_filt) 
SELECT ROW_NUMBER() OVER() AS unique_id, apps.FEIN as rawid, company, {emh.clean_company_string("company")} AS company_clean, n_us_employees AS n, 
    emp_locations, work_locations, top_emp_location, top_work_location,
    naics_codes, naics2_codes, naics3_codes, naics4_codes, naics5_codes, 
    top_naics_code, top_naics2_code, top_naics3_code, top_naics4_code, top_naics5_code
FROM (
    -- 2. collapse all locations by FEIN into list with counts
    (SELECT FEIN, 
        list_transform(
            list_zip(
                array_agg(state_name ORDER BY n_apps_employer_state DESC), 
                array_agg(n_apps_employer_state/n_apps_employer_tot ORDER BY n_apps_employer_state DESC)
            ), 
            x -> struct_pack(state := x[1], share := x[2])
        ) as emp_locations,
        array_agg(state_name ORDER BY n_apps_employer_state DESC).list_extract(1) AS top_emp_location
    FROM (
        -- grouping by FEIN and state
        SELECT FEIN, state_name, MEAN(n_apps_employer_tot) AS n_apps_employer_tot, COUNT(*) AS n_apps_employer_state FROM employer_counts WHERE NOT state_name = 'NA' GROUP BY FEIN, state_name)
    GROUP BY FEIN
    ) AS apps
    FULL JOIN (
    -- 3. collapse all worksites by FEIN into list with counts
    SELECT FEIN, 
        list_transform(
            list_zip(
                array_agg(worksite_state_name ORDER BY n_wins_worksite_state DESC), 
                array_agg(n_wins_worksite_state/n_wins_worksite_tot ORDER BY n_wins_worksite_state DESC)
            ), 
            x -> struct_pack(state := x[1], share := x[2])
        ) as work_locations,
        array_agg(worksite_state_name ORDER BY n_wins_worksite_state DESC).list_extract(1) AS top_work_location
    FROM (
        -- grouping by FEIN and state
        SELECT FEIN, worksite_state_name, MEAN(n_wins_worksite_tot) AS n_wins_worksite_tot, COUNT(*) AS n_wins_worksite_state FROM employer_counts WHERE NOT worksite_state_name = 'NA' GROUP BY FEIN, worksite_state_name)
    GROUP BY FEIN
    ) AS worksites
    ON apps.FEIN = worksites.FEIN
    FULL JOIN (
    -- 4. collapse all NAICS codes by FEIN into list with counts
    {emh.group_by_naics_code_foia()}) AS naics
    ON apps.FEIN = naics.FEIN
    FULL JOIN (
    -- 4a. collapse all NAICS 2-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("2")}) AS naics2
    ON apps.FEIN = naics2.FEIN
    FULL JOIN (
    -- 4b. collapse all NAICS 3-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("3")}) AS naics3
    ON apps.FEIN = naics3.FEIN
    FULL JOIN (
    -- 4c. collapse all NAICS 4-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("4")}) AS naics4
    ON apps.FEIN = naics4.FEIN
    FULL JOIN (
    -- 4d. collapse all NAICS 5-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("5")}) AS naics5
    ON apps.FEIN = naics5.FEIN
    RIGHT JOIN (
    -- 5. main list of employers: by company name and FEIN
    SELECT FEIN, 
        employer_name AS company, MAX(CASE WHEN NOT NUM_OF_EMP_IN_US = 'NA' THEN NUM_OF_EMP_IN_US::INTEGER END) AS n_us_employees,
        COUNT(*) AS n_apps, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_wins,
    FROM foia_filt
    GROUP BY FEIN, company
    ) AS raw
    ON apps.FEIN = raw.FEIN
)
"""
emh.create_replace_table(con,foia_collapse_query, "foia_collapsed", show = False)

## tokenizing both datasets
emh.tokenize(con, "rev_collapsed", "rev_tokenized", "company", "\\s+", show = False)
emh.tokenize(con, "foia_collapsed", "foia_tokenized", "company", "\\s+", show = False)

## to tokenize all data at once:
emh.create_replace_table(con, "SELECT *, 'rev' AS dataset FROM rev_collapsed UNION ALL SELECT *, 'foia' AS dataset FROM foia_collapsed", "all_collapsed")
emh.tokenize(con, "all_collapsed", "all_tokenized", "company", "\\s+", show = True)

## saving main datasets for linking
con.sql("SELECT * FROM foia_tokenized").write_parquet(f"{root}/data/int/splink/foia_tokenized.parquet")
con.sql("SELECT * FROM rev_tokenized").write_parquet(f"{root}/data/int/splink/rev_tokenized.parquet")
con.sql("SELECT * FROM all_tokenized").write_parquet(f"{root}/data/int/splink/all_tokenized.parquet")
