# File Description: Initial Analysis based on R Merge
# Author: Amy Kim
# Date Created: Wed Mar 12

# Imports and Paths
root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

import duckdb as ddb
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import employer_merge_helpers as emh
import analysis_helpers as ah
import statsmodels.formula.api as smf

con = ddb.connect()

# creating sql functions
con.create_function("get_fiscal_year", ah.get_fiscal_year_sql, ["VARCHAR"], "VARCHAR")

con.create_function("get_fiscal_year_foia", ah.get_fiscal_year_foia_sql, ["VARCHAR"], "FLOAT")
    
con.create_function("get_quarter", ah.get_quarter_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_quarter_foia", ah.get_quarter_foia_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_fy_from_quarter", ah.get_fy_from_quarter_sql, ['FLOAT'], 'FLOAT')

# importing data
## duplicate rcids (companies that appear more than once in linkedin data)
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")

## matched company data from R
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")

## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")

## joining raw FOIA data with merged data to get foia_ids in raw foia data
foia_with_ids = con.sql("SELECT *, CASE WHEN matched IS NULL THEN 0 ELSE matched END AS matchind FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id, 1 AS matched FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

# collapsing FOIA raw data to foia_id level
foia_for_merge = con.sql("""
SELECT foia_id, FIRST(employer_name) AS company_FOIA, lottery_year, 
    MAX(CASE WHEN NOT NUM_OF_EMP_IN_US = 'NA' THEN NUM_OF_EMP_IN_US::INTEGER END) AS n_us_employees,
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_apps, 
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 THEN 1 END) AS n_success,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 AND rec_date != 'NA' THEN 1 END) AS n_i129
FROM foia_with_ids WHERE foia_id IS NOT NULL GROUP BY foia_id, lottery_year
""").df()

## revelio data (pre-filtered to only companies in rmerge)
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")

## pre-processed data
merged_for_analysis = con.read_parquet(f"{root}/data/int/merged_for_analysis_mar31.parquet")
balanced_full = con.read_parquet(f"{root}/data/int/balanced_full_mar25.parquet")


#####################################
# DEFINING MAIN SAMPLE
#####################################
# IDing outsourcing/staffing companies
foia_main_samp_unfilt = con.sql("SELECT FEIN, lottery_year, COUNT(CASE WHEN ben_multi_reg_ind = 1 THEN 1 END)/COUNT(*) AS share_multireg, COUNT(*) AS n_apps_tot, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END)/COUNT(*) AS win_rate FROM foia_with_ids GROUP BY FEIN, lottery_year")

n = con.sql('SELECT COUNT(*) FROM foia_main_samp_unfilt').df().iloc[0,0]
print(f"Total Employer x Years: {n}")
print(f"Employer x Years with Fewer than 50 Apps: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE n_apps_tot < 50").df().iloc[0,0]}")
print(f"Employer x Years with Fewer than 50% Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg < 0.5").df().iloc[0,0]}")
print(f"Employer x Years with No Duplicates: {con.sql("SELECT COUNT(*) FROM foia_main_samp_unfilt WHERE share_multireg = 0").df().iloc[0,0]}")

foia_main_samp = con.sql("SELECT * FROM foia_main_samp_unfilt WHERE n_apps_tot < 50 AND share_multireg = 0")
print(f"Preferred Sample: {foia_main_samp.df().shape[0]} ({round(100*foia_main_samp.df().shape[0]/n)}%)")

foia_main_samp_def = con.sql("SELECT *, CASE WHEN n_apps_tot < 50 AND share_multireg = 0 THEN 'insamp' ELSE 'outsamp' END AS sampgroup FROM foia_main_samp_unfilt")
con.sql("SELECT sampgroup, SUM(n_success)/SUM(n_apps_tot) AS total_win_rate FROM foia_main_samp_def GROUP BY sampgroup")

# foia_df = con.sql("SELECT * FROM ((SELECT FEIN, lottery_year, sampgroup FROM foia_main_samp_def) AS a JOIN foia_with_ids AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year)").df()

########################################################
# FIRST STAGE: EFFECT OF WINNING IN t ON NUMBER OF I129s
########################################################
lottery_year = 2023 
fs_joined = con.sql("SELECT a.FEIN AS FEIN, a.lottery_year::FLOAT AS ly, t::FLOAT AS t, n_apps_tot AS n_apps, n_success_tot AS n_wins, n_i129, n_appr, sampgroup FROM (balanced_full AS a JOIN (SELECT FEIN, lottery_year, sampgroup FROM foia_main_samp_def) AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year)")

# single win
fs_collapsed = con.sql(f"SELECT ly, t, n_wins::VARCHAR AS n_wins, MEAN(CASE WHEN n_appr IS NULL THEN 0 ELSE n_appr END) AS mean_appr FROM fs_joined WHERE n_apps = 1 AND sampgroup = 'insamp' AND ly = {lottery_year} AND t < 2024 GROUP BY n_wins, ly, t").df()

print(con.sql(f"SELECT COUNT(DISTINCT FEIN) FROM fs_joined WHERE n_apps = 1 AND sampgroup = 'insamp' AND ly = {lottery_year} "))
#print(fs_collapsed)
pivot_long = fs_collapsed.pivot(index = ['t','ly'], columns = ['n_wins'], values = ['mean_appr'])
pivot_long = pivot_long.join(pivot_long.groupby(level = 0, axis = 1).diff().rename(columns={'1':'diff'}).loc(axis = 1)[:,'diff']).reset_index().melt(id_vars = [('t',''),('ly','')])

pivot_long.columns = ['t', 'ly', 'var', 'treat', 'value']

rawvars = pivot_long.loc[pivot_long['treat'] != 'diff']
diffs = pivot_long.loc[pivot_long['treat'] == 'diff']

fs_plot = sns.scatterplot(data = rawvars, x = 't', y = 'value', hue = 'treat', s = 100)
fs_plot.axvline(x = lottery_year-1.25)
fs_plot.set(xlabel = "Fiscal Year", ylabel = "Avg. Number of H-1B Visas Approved")
fs_plot.legend_.set(title = f"{lottery_year-1} Lottery Wins")
fs_plot

#####################################
# INITIAL ANALYSIS
#####################################
import statsmodels.formula.api as smf

samp_to_foia_id = con.sql("SELECT * FROM ((SELECT FEIN, lottery_year, sampgroup FROM foia_main_samp_def) AS a JOIN (SELECT FEIN, lottery_year, foia_id FROM foia_with_ids GROUP BY FEIN, lottery_year, foia_id) AS b ON a.FEIN = b.FEIN AND a.lottery_year = b.lottery_year) WHERE foia_id IS NOT NULL")

# assumes lottery happens in q2 of previous year -- if q1, change to (t - ly + 1)
out_joined = con.sql(
"""SELECT a.foia_id AS foia_id, main_rcid, 
        a.lottery_year::FLOAT AS ly, 
        t::FLOAT AS t, 
        FLOOR((t - ly + 0.75)::FLOAT) AS t_rel,
        t-ly+0.75 AS q_rel,
        n_apps_tot AS n_apps, 
        n_success_tot AS n_wins, 
        new_positions, 
        new_positions_weighted,
        new_positions_top3,
        new_positions_top10,
        new_positions_highn,
        new_hires, 
        new_hires_weighted,
        new_hires_top3,
        new_hires_top10,
        new_hires_highn,
        n_emp, 
        n_emp_weighted,
        n_emp_top3,
        n_emp_top10,
        n_emp_highn,
        sampgroup,
        MAX(CASE WHEN t > 2019 AND t < ly THEN n_emp ELSE 0 END) OVER(PARTITION BY a.foia_id) AS n_emp_max
    FROM (
        merged_for_analysis AS a 
        JOIN 
        (SELECT foia_id, sampgroup FROM samp_to_foia_id) AS b 
        ON a.foia_id = b.foia_id
    )""")

# regressions
out_for_reg = con.sql("""SELECT ly, main_rcid, q_rel, t, n_wins/n_apps AS win_rate, 
    new_hires, new_positions, n_emp, new_hires_top3, new_hires_top10, new_hires_highn, new_positions_top3, new_positions_top10, new_positions_highn, n_emp_top3, n_emp_top10, n_emp_highn, n_emp_max
    FROM out_joined WHERE t < 2025 AND t > 2014 AND t_rel >= -4 AND t_rel <= 4""").df()
out_for_reg.to_csv(f"{root}/data/int/out_for_reg_apr4.csv")

out_collapsed  = con.sql(
f"""SELECT ly, t_rel, 
        n_wins::VARCHAR AS n_wins, 
        MEAN(CASE WHEN new_hires IS NULL THEN 0 ELSE new_hires END) AS mean_new_hires, 
        MEAN(new_hires) AS mean_new_hires_nulls, 
        MEAN(CASE WHEN new_positions IS NULL THEN 0 ELSE new_positions END) AS mean_new_positions, 
        MEAN(new_positions) AS mean_new_positions_nulls, 
        MEAN(CASE WHEN n_emp IS NULL THEN 0 ELSE n_emp END) AS mean_n_emp, 
        MEAN(n_emp) AS mean_n_emp_nulls, 
        MEAN(CASE WHEN new_hires_weighted IS NULL THEN 0 ELSE new_hires_weighted END) AS mean_new_hires_weighted, 
        MEAN(new_hires_weighted) AS mean_new_hires_nulls_weighted, 
        MEAN(CASE WHEN new_positions_weighted IS NULL THEN 0 ELSE new_positions_weighted END) AS mean_new_positions_weighted, 
        MEAN(new_positions_weighted) AS mean_new_positions_nulls_weighted, 
        MEAN(CASE WHEN n_emp_weighted IS NULL THEN 0 ELSE n_emp_weighted END) AS mean_n_emp_weighted, 
        MEAN(n_emp_weighted) AS mean_n_emp_nulls_weighted,
        MEAN(CASE WHEN new_hires_top3 IS NULL THEN 0 ELSE new_hires_top3 END) AS mean_new_hires_top3,
        MEAN(CASE WHEN new_hires_top10 IS NULL THEN 0 ELSE new_hires_top10 END) AS mean_new_hires_top10,
        MEAN(CASE WHEN new_hires_highn IS NULL THEN 0 ELSE new_hires_highn END) AS mean_new_hires_highn,
        MEAN(CASE WHEN new_positions_top3 IS NULL THEN 0 ELSE new_positions_top3 END) AS mean_new_positions_top3,
        MEAN(CASE WHEN new_positions_top10 IS NULL THEN 0 ELSE new_positions_top10 END) AS mean_new_positions_top10,
        MEAN(CASE WHEN new_positions_highn IS NULL THEN 0 ELSE new_positions_highn END) AS mean_new_positions_highn,
        MEAN(CASE WHEN n_emp_top3 IS NULL THEN 0 ELSE n_emp_top3 END) AS mean_n_emp_top3,
        MEAN(CASE WHEN n_emp_top10 IS NULL THEN 0 ELSE n_emp_top10 END) AS mean_n_emp_top10,
        MEAN(CASE WHEN n_emp_highn IS NULL THEN 0 ELSE n_emp_highn END) AS mean_n_emp_highn
    FROM out_joined 
    WHERE n_apps = 2 AND sampgroup = 'insamp' AND t < 2025 AND t > 2010 AND t_rel > - 8 AND n_emp_max > 10
    GROUP BY n_wins, ly, t_rel
""").df()

lotyear = 2023
print(con.sql(f"SELECT COUNT(DISTINCT foia_id) FROM out_joined WHERE n_apps = 1 AND sampgroup = 'insamp' AND ly = {lotyear} AND n_emp_max > 10"))
#print(fs_collapsed)
print(con.sql(f"SELECT COUNT(*) FROM out_joined WHERE n_apps = 1 AND sampgroup = 'insamp' AND ly = {lotyear}"))
# TODO: GET BALANCED SAMPLE?

pivot_long = out_collapsed.pivot(index = ['t_rel','ly'], columns = ['n_wins'], values = ['mean_new_hires','mean_new_positions','mean_n_emp','mean_new_hires_nulls','mean_new_positions_nulls','mean_n_emp_nulls','mean_new_hires_weighted','mean_new_positions_weighted','mean_n_emp_weighted','mean_new_hires_nulls_weighted','mean_new_positions_nulls_weighted','mean_n_emp_nulls_weighted', 'mean_new_hires_top3', 'mean_new_hires_top10', 'mean_new_hires_highn', 'mean_new_positions_top3', 'mean_new_positions_top10', 'mean_new_positions_highn', 'mean_n_emp_top3', 'mean_n_emp_top10', 'mean_n_emp_highn'])

oneapp = False
if oneapp:
    pivot_long = pivot_long.join(pivot_long.groupby(level = 0, axis = 1).diff().rename(columns={'1':'diff'}).loc(axis = 1)[:,'diff']).reset_index().melt(id_vars = [('t_rel',''),('ly','')])

    pivot_long.columns = ['t_rel', 'ly', 'var', 'treat', 'value']

    rawvars = pivot_long.loc[pivot_long['treat'] != 'diff']
    diffs = pivot_long.loc[pivot_long['treat'] == 'diff']

else:
    pivot_long = pivot_long.reset_index().melt(id_vars = [('t_rel',''),('ly','')])

    pivot_long.columns = ['t_rel', 'ly', 'var', 'treat', 'value']
    rawvars = pivot_long


g = sns.FacetGrid(data = rawvars.loc[(rawvars['var']=='mean_new_positions')&(rawvars['ly'] != 2024)], hue = 'treat', row = 'ly', height = 2, aspect = 4)
g.map(sns.lineplot, 't_rel', 'value').add_legend()
g.refline(x = 0)

g = sns.FacetGrid(data = diffs.loc[diffs['var']=='mean_new_hires_top3'], row = 'ly')
g.map(sns.lineplot, 't_rel', 'value').add_legend()
g.refline(x = 0)


g = sns.FacetGrid(data = diffs.loc[diffs['var']=='mean_new_positions_weighted'], row = 'ly')
g.map(sns.lineplot, 't_rel', 'value').add_legend()
g.refline(x = 0)


g = sns.FacetGrid(data = diffs.loc[diffs['var']=='mean_n_emp_weighted'], row = 'ly')
g.map(sns.lineplot, 't_rel', 'value').add_legend()
g.refline(x = 0)
# out = sns.lineplot(data = rawvars.loc[rawvars['var']=='mean_new_positions'], x = 't', y = 'value', hue = 'treat')
# out.axvline(x = lotyear-0.75)
# out.set(xlabel = "Quarter of Start Date", ylabel = "Avg. Number of New Hires")
# out.legend_.set(title = f"{lotyear-1} Lottery Wins")

# plt.figure()
# out2 = sns.lineplot(data = rawvars.loc[rawvars['var']=='mean_n_emp'], x = 't', y = 'value', hue = 'treat')
# out2.axvline(x = lotyear-0.75)
# out2.set(xlabel = "Quarter of Start Date", ylabel = "Avg. Number of Employees")
# out2.legend_.set(title = f"{lotyear - 1} Lottery Wins")