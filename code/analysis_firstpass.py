# File Description: Initial Analysis based on R Merge
# Author: Amy Kim
# Date Created: Wed Mar 12

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import fiscalyear
import seaborn as sns
import employer_merge_helpers as emh
import analysis_helpers as ah

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()

# creating sql functions
con.create_function("get_fiscal_year", ah.get_fiscal_year_sql, ["VARCHAR"], "VARCHAR")

con.create_function("get_fiscal_year_foia", ah.get_fiscal_year_foia_sql, ["VARCHAR"], "FLOAT")
    
con.create_function("get_quarter", ah.get_quarter_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_quarter_foia", ah.get_quarter_foia_sql, ["VARCHAR"], "FLOAT")

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
con.sql("CREATE OR REPLACE TABLE foia_with_ids AS SELECT foia_id, employer_name, NUM_OF_EMP_IN_US, a.FEIN AS FEIN, status_type, ben_multi_reg_ind, rec_date, first_decision_date, FIRST_DECISION, a.lottery_year AS lottery_year FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

## merged revelio data
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")
con.sql("CREATE OR REPLACE TABLE merged_pos AS SELECT * FROM merged_pos")


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
SELECT foia_id, FIRST(employer_name) AS company_FOIA, lottery_year,
    MAX(CASE WHEN NOT NUM_OF_EMP_IN_US = 'NA' THEN NUM_OF_EMP_IN_US::INTEGER END) AS n_us_employees,
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_apps, 
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 THEN 1 END) AS n_success,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 AND rec_date != 'NA' THEN 1 END) AS n_i129
FROM foia_with_ids WHERE foia_id IS NOT NULL GROUP BY foia_id, lottery_year
"""
emh.create_replace_table(con = con, query = query2, table_out="foia_for_merge", show = False)

# STEP THREE: clean and collapse revelio data to main_rcid level (by fiscal year of start date)
query3 = """
SELECT main_rcid, startfy AS t,
    COUNT(*) AS new_positions,
    COUNT(CASE WHEN rank = 1 THEN 1 END) AS new_hires
FROM 
    -- rank positions by start date within user id x main_rcid and get FY of start date
    (SELECT ROW_NUMBER() OVER(PARTITION BY main_rcid, user_id ORDER BY get_fiscal_year(startdate)) AS rank,
        get_fiscal_year(startdate) AS startfy,
        country, main_rcid 
    FROM (
        SELECT startdate, user_id, country,
            CASE WHEN main_rcid IS NULL THEN pos.rcid ELSE main_rcid END AS main_rcid
        FROM (merged_pos AS pos
        LEFT JOIN 
            dup_rcids AS dup_cw
        ON pos.rcid = dup_cw.rcid)
        )
    )
WHERE country = 'United States'
GROUP BY main_rcid, startfy
"""
emh.create_replace_table(con = con, query = query3, table_out="rev_for_merge", show = False)

# STEP THREE POINT FIVE: clean and collapse revelio data to main_rcid level (by quarter of start date)
query3_5 = """
SELECT main_rcid, startq AS t,
    COUNT(*) AS new_positions,
    COUNT(CASE WHEN rank = 1 THEN 1 END) AS new_hires
FROM 
    -- rank positions by start date within user id x main_rcid and get quarter of start date
    (SELECT ROW_NUMBER() OVER(PARTITION BY main_rcid, user_id ORDER BY get_quarter(startdate)) AS rank,
        get_quarter(startdate) AS startq,
        country, main_rcid 
    FROM (
        SELECT startdate, user_id, country,
            CASE WHEN main_rcid IS NULL THEN pos.rcid ELSE main_rcid END AS main_rcid
        FROM (merged_pos AS pos
        LEFT JOIN 
            dup_rcids AS dup_cw
        ON pos.rcid = dup_cw.rcid)
        )
    )
WHERE country = 'United States'
GROUP BY main_rcid, startq
"""
emh.create_replace_table(con = con, query = query3_5, table_out="rev_for_merge_q", show = False)

# STEP FOUR: merge all datasets
merge_query = """
SELECT lottery_year, ids.main_rcid AS main_rcid, ids.foia_id AS foia_id, company_FOIA, n_us_employees, n_apps_tot, n_apps, n_success_tot, n_success, n_i129, t, new_positions, new_hires FROM (
    id_merge AS ids 
    JOIN 
    foia_for_merge AS foia
    ON ids.foia_id = foia.foia_id
    LEFT JOIN
    rev_for_merge AS rev
    ON ids.main_rcid = rev.main_rcid
)
WHERE t IS NOT NULL
"""
emh.create_replace_table(con = con, query = merge_query, table_out="merged_for_analysis", show = False)

# STEP FOUR POINT FIVE: merge all datasets (Q level, not FY)
merge_queryq = """
SELECT lottery_year, ids.main_rcid AS main_rcid, ids.foia_id AS foia_id, company_FOIA, n_us_employees, n_apps_tot, n_apps, n_success_tot, n_success, n_i129,  t, new_positions, new_hires FROM (
    id_merge AS ids 
    JOIN 
    foia_for_merge AS foia
    ON ids.foia_id = foia.foia_id
    LEFT JOIN
    rev_for_merge_q AS rev
    ON ids.main_rcid = rev.main_rcid
)
WHERE t IS NOT NULL
"""
emh.create_replace_table(con = con, query = merge_queryq, table_out="merged_for_analysisq", show = False)

########################################################
# FIRST STAGE: EFFECT OF WINNING IN t ON NUMBER OF I129s
########################################################
# idea: for balanced panel of firms (observed in foia data in all years), plot mean number of i129s submitted in each FY by n_success
ly = 2023
p = 0.269

# step one: get balanced panel of firms (observed in all four years, drop any duplicate rcids or feins for now)
balanced_panel = con.sql("SELECT lottery_year, foia_id, FEIN FROM (SELECT FEIN, foia_id, lottery_year, COUNT(DISTINCT lottery_year) OVER(PARTITION BY FEIN) AS n_lot_years, COUNT(DISTINCT main_rcid) OVER(PARTITION BY FEIN) AS n_main_rcids FROM (SELECT *, COUNT(DISTINCT FEIN) OVER (PARTITION BY foia_id) AS n_feins FROM rmerge) WHERE n_feins = 1) WHERE n_lot_years = 4 AND n_main_rcids = 1 GROUP BY lottery_year, foia_id, FEIN")

# step two: merge with raw foia data 
balanced_samp = con.sql("SELECT *, get_fiscal_year_foia(rec_date) AS rec_fy, get_fiscal_year_foia(first_decision_date) AS dec_fy, get_quarter_foia(rec_date) AS rec_q, get_quarter_foia(first_decision_date) AS dec_q FROM (foia_with_ids AS a JOIN balanced_panel AS b ON a.foia_id = b.foia_id)")

# step three: restrict to ly to get n apps and successes (for treatment variable and sample restriction) at the FEIN level [note: can't use foia_for_merge because it's collapsed at the foia_id level not FEIN x lottery_year]
samp_treat = con.sql(
f"""
SELECT FEIN, 
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_apps, 
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 THEN 1 END) AS n_success
FROM balanced_samp 
WHERE lottery_year = {ly} 
GROUP BY FEIN
""")

# step four: group raw foia data by fiscal year received to get number of i129s received by fiscal year
foia_i129_rec = con.sql(
"""
SELECT FEIN, rec_fy, 
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_i129
FROM balanced_samp
WHERE rec_date != 'NA'
GROUP BY FEIN, rec_fy
""")

foia_approvals = con.sql(
"""
SELECT FEIN, dec_fy, 
    COUNT(CASE WHEN ben_multi_reg_ind = 0 AND FIRST_DECISION = 'Approved' THEN 1 END) AS n_appr
FROM balanced_samp
WHERE first_decision_date != 'NA'
GROUP BY FEIN, dec_fy
""")

# step four point five: do this by quarter
foia_i129_recq = con.sql(
"""
SELECT FEIN, rec_q, 
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_i129
FROM balanced_samp
WHERE rec_date != 'NA'
GROUP BY FEIN, rec_q
""")

# step five: merge together (left join onto cross join of foia_i129_rec rec_fys and FEINs to code missing n_i129s as 0s)
fs_full = con.sql(
f"""
SELECT *, n_success - (n_apps*{p}) AS U FROM (
    (SELECT * FROM (SELECT rec_fy FROM foia_i129_rec WHERE rec_fy != 2019 GROUP BY rec_fy) CROSS JOIN (SELECT FEIN FROM balanced_panel GROUP BY FEIN)) AS a 
    LEFT JOIN 
    samp_treat AS b
    ON a.FEIN = b.FEIN
    LEFT JOIN 
    foia_i129_rec AS c
    ON a.rec_fy = c.rec_fy AND a.FEIN = c.FEIN
)
"""
)

fs_full_approvals = con.sql(
f"""
SELECT *, n_success - (n_apps*{p}) AS U FROM (
    (SELECT * FROM (SELECT dec_fy FROM foia_approvals WHERE dec_fy != 2019 GROUP BY dec_fy) CROSS JOIN (SELECT FEIN FROM balanced_panel GROUP BY FEIN)) AS a 
    LEFT JOIN 
    samp_treat AS b
    ON a.FEIN = b.FEIN
    LEFT JOIN 
    foia_approvals AS c
    ON a.dec_fy = c.dec_fy AND a.FEIN = c.FEIN
)
"""
)


fs_fullq = con.sql(
"""
SELECT *, n_success - (n_apps*{p}) AS U FROM (
    (SELECT * FROM (SELECT rec_q FROM foia_i129_recq WHERE rec_q >= 2020 GROUP BY rec_q) CROSS JOIN (SELECT FEIN FROM balanced_panel GROUP BY FEIN)) AS a 
    LEFT JOIN 
    samp_treat AS b
    ON a.FEIN = b.FEIN
    LEFT JOIN 
    foia_i129_recq AS c
    ON a.rec_q = c.rec_q AND a.FEIN = c.FEIN
)
"""
)

fs_collapsed = con.sql("SELECT rec_fy::FLOAT AS t, n_success::VARCHAR AS treat, MEAN(CASE WHEN n_i129 IS NULL THEN 0 ELSE n_i129 END) AS mean_y, SUM(CASE WHEN n_i129 IS NULL THEN 0 ELSE n_i129 END) AS sum_y, FROM fs_full WHERE n_apps = 1 GROUP BY n_success, rec_fy").df()

(fs_rawvars, fs_diffs) = ah.bin_long_diffs(fs_collapsed, ["mean_y"])
ah.graph_df_hue(fs_rawvars, ly, 'treat').set(ylabel = "Number of H1B Visa Forms Submitted")
ah.graph_df_nohue(fs_diffs, ly)

fs_approvals = con.sql("SELECT dec_fy::FLOAT AS t, n_success::VARCHAR AS treat, MEAN(CASE WHEN n_appr IS NULL THEN 0 ELSE n_appr END) AS mean_y, SUM(CASE WHEN n_appr IS NULL THEN 0 ELSE n_appr END) AS sum_y, FROM fs_full_approvals WHERE n_apps = 1 GROUP BY n_success, dec_fy").df()

(fs_rawvars_appr, fs_diffs_appr) = ah.bin_long_diffs(fs_approvals, ["mean_y"])
ah.graph_df_hue(fs_rawvars_appr, ly, 'treat').set(ylabel = "Number of H1B Visas Initially Approved")
ah.graph_df_nohue(fs_diffs_appr, ly)

fs_collapsedq = con.sql("SELECT rec_q::FLOAT AS t, n_success::VARCHAR AS treat, MEAN(CASE WHEN n_i129 IS NULL THEN 0 ELSE n_i129 END) AS mean_y, SUM(CASE WHEN n_i129 IS NULL THEN 0 ELSE n_i129 END) AS sum_y, FROM fs_fullq WHERE n_apps = 1 GROUP BY n_success, rec_q").df()

(fs_rawvarsq, fs_diffsq) = bin_long_diffs(fs_collapsedq, ["mean_y", "sum_y"])
ah.graph_df_hue(fs_rawvarsq, ly, 'treat')
ah.graph_df_nohue(fs_diffsq, ly)


fs_collapsed_full = con.sql("SELECT rec_fy::FLOAT AS t, n_success::VARCHAR AS treat, MEAN(CASE WHEN n_i129 IS NULL THEN 0 ELSE n_i129 END) AS mean_y, SUM(CASE WHEN n_i129 IS NULL THEN 0 ELSE n_i129 END) AS sum_y, FROM fs_full WHERE n_apps < 5 GROUP BY n_success, rec_fy").df()

fs_full_rawvars = fs_collapsed_full.melt(id_vars=['t','treat'])
fs_full_rawvars.columns = ['t','treat','var','value']

ah.graph_df_hue(fs_full_rawvars, ly, 'treat')

# continuous
fs_all_fy2022 = con.sql("SELECT U, n_i129 FROM fs_full WHERE rec_fy = 2022").df() 
sns.regplot(data = fs_all_fy2022, x = "U", y = "n_i129", x_bins = 20)

#####################################
# INITIAL ANALYSIS
#####################################
# to change time period from here, just change 't_unit' to ''/'q'
t_unit = ''
lottery_year = 2021
yvar = 'new_hires'
treatvar = 'n_success'
bin_treat = True

# defining sample for analysis
analysis = con.sql(f"SELECT t::FLOAT AS t, {yvar} AS y, {treatvar}::VARCHAR AS treat, SUM(new_hires) OVER(PARTITION BY main_rcid) AS n_hires_tot FROM merged_for_analysis{t_unit} WHERE t::FLOAT > 2018 AND t::FLOAT < 2025 AND lottery_year = {lottery_year} AND n_apps = 1")

analysisdf = analysis.df()

analysis_cont = con.sql(f"SELECT t::FLOAT AS t, new_hires, n_success - (n_apps*{p}) AS U, n_apps AS n, SUM(new_hires) OVER(PARTITION BY main_rcid) AS n_hires_tot FROM merged_for_analysis WHERE t::FLOAT = {lottery_year - 1} AND lottery_year = {lottery_year}")

analysis_contdf = analysis_cont.df()

sns.regplot(data = analysis_contdf.loc[(analysis_contdf['new_hires'] < 1000)&(analysis_contdf['n'] < 10)], x = "U", y = "new_hires", x_bins = 50)

# collapsing by treatment
collapsed = con.sql("SELECT MEAN(y) AS mean_y, MEDIAN(y) AS median_y, SUM(y)/COUNT(y) AS wmean_y, VARIANCE(y) AS var_y, t, treat FROM analysis GROUP BY treat, t").df()

# for binary treatment: pivot long and get diffs
varlist = ['mean_y', 'var_y']# ['mean_y', 'median_y', 'wmean_y', 'var_y']
if bin_treat:
    pivot_long = collapsed.pivot(index = 't', columns = ['treat'], values = varlist)
    pivot_long = pivot_long.join(pivot_long.groupby(level=0,axis=1).diff().rename(columns={'1':'diff'}).loc(axis = 1)[:,'diff']).reset_index().melt(id_vars = [('t','')])

    pivot_long.columns = ['t', 'var', 'treat', 'value']

    rawvars = pivot_long.loc[pivot_long['treat'] != 'diff']
    diffs = pivot_long.loc[pivot_long['treat'] == 'diff']

else:
    rawvars = collapsed.melt(id_vars = 't')

# g = sns.FacetGrid(data = rawvars, row = 'var', hue = 'treat', sharey = False, height = 2, aspect = 2)
# g.map(sns.scatterplot, 't', 'value')
# g.refline(x = lottery_year - 0.75)

g_diff = sns.FacetGrid(data = diffs, row = 'var', sharey = False, height = 2, aspect = 2)
g_diff.map(sns.scatterplot, 't', 'value')
g_diff.refline(x = lottery_year - 0.75)
# for var in pivot_long['var'].unique():

# sns.scatterplot(x = 't', y = 'value', data = pivot_long.loc[(pivot_long['n_success']=='diff') & (pivot_long['var'] == 'mean_hires')]).axvline(2022.5)

# sns.scatterplot(x = 'startfy', y = 'value', hue = 'n_success', data = pivot_long.loc[(pivot_long['n_success']!='diff') & (pivot_long['var'] == 'mean_hires')]).axvline(2022.5)

