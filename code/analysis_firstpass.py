# File Description: Initial Analysis based on R Merge
# Author: Amy Kim
# Date Created: Wed Mar 12

# Imports and Paths
import duckdb as ddb
import time
import pandas as pd
import random
import fiscalyear
import employer_merge_helpers as emh

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

## HELPERS
# get fiscal year from date (pandas)
def get_fiscal_year(date):
    if pd.isnull(date):
        return None
    return fiscalyear.FiscalDate(date.year, date.month, date.day).fiscal_year

# get fiscal year from date (sql)
def get_fiscal_year_sql(date):
    month = int(date[5:7])
    year = int(date[:4])
    if month >= 10:
        return year + 1
    else:
        return year 
    
con.create_function("get_fiscal_year", get_fiscal_year_sql, ["VARCHAR"], "VARCHAR")
    
#####################
# IMPORTING DATA
#####################
## duplicate rcids
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")
con.sql("CREATE OR REPLACE TABLE dup_rcids AS SELECT main_rcid, rcid FROM dup_rcids, GROUP BY main_rcid, rcid")

## Importing R Matched data
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")
con.sql("CREATE OR REPLACE TABLE rmerge AS SELECT * FROM rmerge")

## raw foia data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")
# merging with rmerge to get foia_ids (drop missings and multi_reg_ind = 1 ???) ## TODO: figure out how to deal w multi_reg
con.sql("CREATE TABLE foia_with_ids AS SELECT * FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

## merged revelio data
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")
con.sql("CREATE OR REPLACE TABLE merged_pos AS SELECT * FROM merged_pos")

#####################
# EXPLORING DATA
#####################
samp = con.sql("SELECT * FROM merged_pos WHERE rcid IN (3664186, 222572, 221338, 96346146, 3081301)").df()
samp['fy'] = pd.to_datetime(samp['startdate']).apply(get_fiscal_year).apply(lambda x: None if pd.isnull(x) else x)

# getting positions with users new at the company
samp_new = con.sql("SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY rcid, user_id ORDER BY startdate) AS rank FROM merged_pos WHERE rcid IN (3664186, 222572, 221338, 96346146, 3081301)) WHERE rank = 1").df()
samp_new['fy'] = pd.to_datetime(samp_new['startdate']).apply(get_fiscal_year).apply(lambda x: None if pd.isnull(x) else x)

# get number of positions by starting fiscal year and plot by rcid
plotdata = samp.loc[(samp['fy'] > 2015) & ((samp['rcid']==3664186) | (samp['rcid'] == 222572))].groupby(['fy','rcid']).size().reset_index(name='count')

plotdata_new = samp_new.loc[(samp_new['fy'] > 2015) & ((samp_new['rcid']==3664186) | (samp_new['rcid'] == 222572))].groupby(['fy','rcid']).size().reset_index(name='count')

plotdata['rcidcol'] = plotdata['rcid'].astype("category")

plotdata_new['rcidcol'] = plotdata_new['rcid'].astype("category")

plotdata.plot.scatter(x='fy',y='count',c='rcidcol', cmap = "viridis", s = 50).axvline(2020.5)
plotdata_new.plot.scatter(x='fy',y='count',c='rcidcol', cmap = "viridis", s = 50).axvline(2020.5)


#####################################
# COLLAPSING DATA FOR ANALYSIS
#####################################
# STEP ONE: get central crosswalk of matched IDs unique at the main_rcid x FEIN x lottery_year level (note: RMERGE is unique at the FEIN x year level (no duplicate main_RCIDs), but is not unique at the main_rcid x year level (may have duplicate FEINs), so need to collapse)
query1 = """
-- collapse matched IDs to be unique at the rcid x FEIN x lottery_year level
SELECT lottery_year, main_rcid, foia_id
FROM rmerge 
GROUP BY foia_id, lottery_year, main_rcid
"""
emh.create_replace_table(con = con, query = query1, table_out="id_merge", show = False)

# STEP TWO: clean and collapse FOIA raw data to foia_id level
query2 = """
SELECT foia_id, FIRST(employer_name) AS company_FOIA,
    MAX(CASE WHEN NOT NUM_OF_EMP_IN_US = 'NA' THEN NUM_OF_EMP_IN_US::INTEGER END) AS n_us_employees,
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_apps, 
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 THEN 1 END) AS n_success
FROM foia_with_ids WHERE foia_id IS NOT NULL GROUP BY foia_id
"""
emh.create_replace_table(con = con, query = query2, table_out="foia_for_merge", show = False)

# STEP THREE: clean and collapse revelio data to main_rcid level
query3 = """
SELECT main_rcid, startfy,
    COUNT(*) AS new_positions,
    COUNT(CASE WHEN rank = 1 THEN 1 END) AS new_hires
FROM 
    -- rank positions by start date within user id x main_rcid and get FY of start date
    (SELECT ROW_NUMBER() OVER(PARTITION BY main_rcid, user_id ORDER BY startdate) AS rank,
        get_fiscal_year(startdate) AS startfy,
        country, main_rcid 
    FROM (
        SELECT startdate, user_id, country,
            CASE WHEN main_rcid IS NULL THEN main.rcid ELSE main_rcid END AS main_rcid
        FROM (merged_pos AS main 
        LEFT JOIN 
            dup_rcids AS dup_cw
        ON main.rcid = dup_cw.rcid)
        )
    )
WHERE country = 'United States'
GROUP BY main_rcid, startfy
"""
emh.create_replace_table(con = con, query = query3, table_out="rev_for_merge", show = False)

# STEP FOUR: merge all datasets
merge_query = """
SELECT lottery_year, ids.main_rcid AS main_rcid, ids.foia_id AS foia_id, company_FOIA, n_us_employees, n_apps_tot, n_apps, n_success_tot, n_success, startfy, new_positions, new_hires FROM (
    id_merge AS ids 
    JOIN 
    foia_for_merge AS foia
    ON ids.foia_id = foia.foia_id
    LEFT JOIN
    rev_for_merge AS rev
    ON ids.main_rcid = rev.main_rcid
)
"""
emh.create_replace_table(con = con, query = merge_query, table_out="merged_for_analysis", show = False)

#####################################
# INITIAL ANALYSIS
#####################################
analysis = con.sql("SELECT * FROM merged_for_analysis WHERE startfy::BIGINT > 2015 AND startfy::BIGINT < 2025 AND lottery_year = 2023 AND n_apps = 1")

collapsed = con.sql("SELECT MEAN(new_hires) AS mean_hires, MEDIAN(new_hires) AS median_hires, startfy, n_success FROM analysis WHERE startfy IS NOT NULL GROUP BY n_success, startfy").df()

collapsed['startfy'] = collapsed['startfy'].apply(int)

collapsed.plot.scatter(x='startfy',y='mean_hires',c='n_success', cmap = "viridis", s = 50).axvline(2022.5)