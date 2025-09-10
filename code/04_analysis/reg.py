# File Description: First Pass at Analysis
# Author: Amy Kim
# Date Created: Jul 23 2025

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import sys
from linearmodels.panel import PanelOLS
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import seaborn as sns
from linearmodels.panel import compare
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# get merge functions from main file
sys.path.append('03_indiv_merge/')
from indiv_merge import merge, merge_df

con = ddb.connect()

#####################
# IMPORTING DATA
#####################
## foia
foia_indiv = con.read_parquet(f'{root}/data/clean/foia_indiv.parquet')

## revelio
rev_indiv = con.read_parquet(f'{root}/data/clean/rev_indiv.parquet')

## revelio user x position level data
rev_pos = con.read_parquet(f"{root}/data/int/wrds_positions_aug1.parquet")

## revelio education data
rev_educ = con.read_parquet(f'{root}/data/int/rev_educ_long_aug8.parquet')

# collapsing to user x institution (for now use country with top score)
con.sql("CREATE OR REPLACE TABLE rev_educ_clean AS SELECT *, ed_startdate AS startdate, ed_enddate AS enddate FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id, education_number ORDER BY matchscore DESC) AS match_order FROM rev_educ WHERE degree_clean != 'Non-Degree') WHERE match_order = 1")

# Importing User x Position-level Data (all positions, cleaned and deduplicated)
merged_pos = con.read_parquet(f'{root}/data/int/merged_pos_clean_aug8.parquet')

# removing duplicates, setting alt enddate as enddate if missing
con.sql("CREATE OR REPLACE TABLE merged_pos_clean AS SELECT * EXCLUDE (enddate), CASE WHEN alt_enddate IS NULL THEN enddate ELSE alt_enddate END AS enddate FROM merged_pos WHERE pos_dup_ind IS NULL OR pos_dup_ind = 0")

#all_merge_filts = [merge_filt_base, merge_filt_prefilt, merge_filt_postfilt]

#mergedf = merge_df(with_t_vars = True, postfilt = 'indiv', MATCH_MULT_CUTOFF = 2)
#####################
# CLEANING DATA
#####################
def merge_filt_clean(merge_raw, index = 'firm x year'):
    ## DEFINING VARIABLES
    merge_filt = merge_raw.copy() 

    # indicator for winning lottery
    merge_filt['winner'] = merge_filt['status_type'] == "SELECTED"

    # age
    merge_filt['age'] = merge_filt['lottery_year'].astype('int') - merge_filt['yob'].astype('int')

    # indicator for valid ADE
    merge_filt['ade'] = np.where((merge_filt['ade_ind'].isnull() == 0) & (merge_filt['ade_ind'] == 1), np.where((merge_filt['ade_year'].isnull()) | (merge_filt['ade_year'] <= merge_filt['lottery_year'].astype('int') - 1), 1, 0), 0)

    # time since last grad year
    merge_filt['graddiff'] = merge_filt['lottery_year'].astype('int') - 1 - merge_filt['last_grad_year']
    merge_filt["graddiff_agg"] = (
    merge_filt
    .groupby("foia_indiv_id")
    .apply(lambda g: (
        (g.dropna(subset=["graddiff", "weight_norm"])
           .assign(weighted=lambda x: x["graddiff"] * x["weight_norm"])
           .pipe(lambda x: x["weighted"].sum() / x["weight_norm"].sum()))
    ))
    .reindex(merge_filt["foia_indiv_id"])
    .values
    )
#     merge_filt['graddiff_agg'] = merge_filt.groupby('foia_indiv_id')['graddiff'].agg('graddiff_agg' : lambda val: np.average(val, weights = merge_filt.loc[x.index, 'weight_norm']))
# df = merge_out.groupby(['foia_indiv_id', 'FEIN', 'female_ind', 'yob', 'age', 'lottery_year', 'foia_country', 'winner', 'n_apps', 'n_unique_country', 'high_rep_emp_ind', 'no_rep_emp_ind']).agg(
#         **{var: (var, lambda x, var = var: np.average(x, weights = merge_out.loc[x.index, 'weight_norm'])) for var in outcomes}
#     ).reset_index()

    # indicator for still working at firm n years later
    merge_filt['work_1yr'] = np.where(merge_filt['enddatediff'] >= 12, 1, 0)
    merge_filt['work_2yr'] = np.where(merge_filt['enddatediff'] >= 24, 1, 0)
    merge_filt['work_3yr'] = np.where(merge_filt['lottery_year'] == '2024', np.nan, np.where(merge_filt['enddatediff'] >= 36, 1, 0)) 

    # promotion
    merge_filt['promote1'] = np.where((merge_filt['change_position1'] == 1) & (merge_filt['change_company1'] == 0), 1, 0)
    merge_filt['promote2'] = np.where((merge_filt['change_position2'] == 1) & (merge_filt['change_company1'] == 0), 1, 0)

    # setting index
    merge_filt['firm_year_fe'] = merge_filt['FEIN'].astype(str) + '_' + merge_filt['lottery_year'].astype(str)
    merge_filt['lottery_year'] = merge_filt['lottery_year'].astype(int)
    merge_filt['year'] = merge_filt['lottery_year'].astype('int')
    merge_filt['emp_id'] = merge_filt['FEIN']

    if index == 'firm + year':
        return merge_filt.set_index(['emp_id', 'year'])
    
    elif index == 'firm x year':
        return merge_filt.set_index(['firm_year_fe', 'foia_indiv_id'])
    
    return merge_filt

# collapsing merged data to application level
def merge_collapse(merge_clean, index = 'firm x year'):
    merge_out = merge_clean.copy().reset_index()
    
    ## COLLAPSING TO APPLICATION LEVEL
    outcomes = ['change_company1', 'change_company2','change_company3', 'promote1', 'promote2', 'in_us1', 'in_us2', 'ade', 'work_1yr', 'work_2yr', 'new_educ1', 'new_educ2', 'agg_compensation1', 'agg_compensation2', 'graddiff', 'in_home_country1', 'in_home_country2', 'loc_null1', 'loc_null2']
    df = merge_out.groupby(['foia_indiv_id', 'FEIN', 'female_ind', 'yob', 'age', 'lottery_year', 'foia_country', 'winner', 'n_apps', 'n_unique_country', 'high_rep_emp_ind', 'no_rep_emp_ind']).agg(
        **{var: (var, lambda x, var = var: np.average(x, weights = merge_out.loc[x.index, 'weight_norm'])) for var in outcomes}
    ).reset_index()

    ## MORE CLEANING
    # year, fein, const
    df['const'] = 1
    df['firm_year_fe'] = df['FEIN'].astype(str) + '_' + df['lottery_year'].astype(str)
    df['lottery_year'] = df['lottery_year'].astype(int)
    df['year'] = df['lottery_year'].astype('int')
    df['emp_id'] = df['FEIN']

    if index == 'firm + year':
        return df.set_index(['emp_id', 'year'])
    
    elif index == 'firm x year':
        return df.set_index(['firm_year_fe', 'foia_indiv_id'])

    else:
        return df

# mergedfs_raw = [pd.read_parquet(f'{root}/data/int/merge_filt_mult{c}_sep8.parquet') for c in [2,4,6]] + [pd.read_parquet(f'{root}/data/int/merge_filt_baseline_sep8.parquet')]
mergedfs_raw = [pd.read_parquet(f'{root}/data/int/merge_filt_mult{c}_sep8.parquet') for c in [2,4,6]]

mergedf_prefilt_raw = pd.read_parquet(f'{root}/data/int/merge_filt_prefilt_aug13.parquet')

mergedfs_clean = [merge_filt_clean(df) for df in mergedfs_raw]
mergedfs = [merge_collapse(df, 'firm x year') for df in mergedfs_clean]
mergedf_clean = merge_filt_clean(mergedf_prefilt_raw)
mergedf = merge_collapse(mergedf_clean, 'firm x year')
# foia_tab = con.sql("CREATE OR REPLACE TABLE foia AS SELECT * FROM foia_indiv USING SAMPLE 100")
# rev_tab = con.sql("CREATE OR REPLACE TABLE rev AS SELECT * FROM rev_indiv")
# mergetest_raw = merge_df(rev_tab = 'rev', foia_tab = 'foia', with_t_vars=True, con = con)
# mergetest = merge_filt_clean(mergetest_raw)

#####################
# BALANCE TESTS
#####################
# avg mean match weight by win
for mergedf in mergedfs_raw:
    mergedf['winner'] = mergedf['status_type'] == "SELECTED"
    # print(mergedf.groupby(['foia_indiv_id','winner'])['weight_norm'].agg('max').reset_index().groupby('winner')['weight_norm'].agg('mean'))
    print("mean match weight")
    print(mergedf.groupby(['foia_indiv_id','winner'])['weight_norm'].agg('mean').reset_index().groupby('winner')['weight_norm'].agg('mean'))
    
    print("mean multiplicity")
    print(mergedf.groupby(['foia_indiv_id','winner'])['weight_norm'].size().reset_index().groupby('winner')['weight_norm'].agg('mean'))
    
# foia_dfs = [foia_raw.assign(dfname = 'All FOIA Apps'), foia_indiv.assign(dfname = 'Apps in Samp'), merge_filt_clean(merge_filt_base, '').assign(dfname = 'Matched Apps'), merge_filt_clean(merge_filt_prefilt, '').assign(dfname = 'Matched Excluding India/China+'), merge_filt_clean(merge_filt_postfilt, '').assign(dfname = 'Matched Post-Filtered')]
# foia_dfs = [foia_indiv.assign(dfname = '0. Apps in Samp').assign(winner = np.where(foia_indiv['status_type'] == 'SELECTED', 1, 0)), all_dfs[0].assign(dfname = '1. Baseline'), all_dfs[1].assign(dfname = '2. Pre-Filtered'), all_dfs[2].assign(dfname = '3. Post-Filtered')]

# foia_dfs_concat = pd.concat(foia_dfs)
# foia_dfs_concat['india'] = np.where(foia_dfs_concat['foia_country'] == 'India', 1, 0)
# foia_dfs_concat['china'] = np.where(foia_dfs_concat['foia_country'] == 'China', 1, 0)
# foia_dfs_concat['age'] = foia_dfs_concat['lottery_year'].astype('int') - foia_dfs_concat['yob'].astype('int')

# print(foia_dfs_concat.groupby('dfname')[['winner','female_ind', 'age', 'india', 'china', 'ade']].agg(lambda x: round(x.mean(), 2)).join(foia_dfs_concat.groupby(['dfname','FEIN', 'lottery_year'])[['n_apps','n_unique_country']].agg('mean').reset_index().groupby('dfname')[['n_apps','n_unique_country']].agg(lambda x: round(x.mean(), 2))).join(foia_dfs_concat.groupby('dfname').size().rename('n')).T)

# formatting 
def panelols_to_latex(results, col_labels, row_var = 'winner', verbose = False):
    """
    results   : list of PanelOLS results objects
    col_labels: list of column headers (e.g. ["us1yr","us2yr",...])
    row_var   : str, the variable name to display (e.g. "win")
    """
    if verbose:
        dict_print = {}
        for i in range(len(results)):
            dict_print[col_labels[i]] = results[i]

        print(compare(dict_print, stars = True, precision = 'std_errors'))

    
    # --- Header ---
    latex = "\\begin{tabular}{l" + "c"*len(results) + "}\n"
    latex += "\\toprule\n"
    
    # Column labels
    latex += "& " + " & ".join(col_labels) + " \\\\\n"
    latex += "& (" + ") & (".join([str(i) for i in range(1,len(results) + 1)]) + ") \\\\\n"
    latex += "\\midrule\n"
    
    # --- Coefficients & SEs ---
    coefs = []
    ses = []
    for res in results:
        b = res.params[row_var]
        se = res.std_errors[row_var]
        stars = ""
        if res.pvalues[row_var] < 0.01:
            stars = "***"
        elif res.pvalues[row_var] < 0.05:
            stars = "**"
        elif res.pvalues[row_var] < 0.1:
            stars = "*"
        coefs.append(f"{b:.4f}{stars}")
        ses.append(f"({se:.4f})")
    
    latex += row_var + " & " + " & ".join(coefs) + " \\\\\n"
    latex += "& " + " & ".join(ses) + " \\\\\n"
    
    # --- Obs ---
    latex += "\\midrule\n"
    latex += "\\textbf{Obs.} & " + " & ".join(str(int(res.nobs)) for res in results) + " \\\\\n"
    
    # --- DV mean ---
    dv_means = []
    for res in results:
        dv_series = res.model.dependent.dataframe.squeeze()
        dv_means.append(dv_series.mean())
    latex += "\\midrule\n"
    latex += "\\textbf{DV mean} & " + " & ".join(f"{m:.3f}" for m in dv_means) + " \\\\\n"
    
    
    # --- Footer ---
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}"
    
    return latex


#####################
# REGRESSIONS
#####################
m0_0s, m0_1s, m1s, m2s, m3s = [], [], [], [], []
wls = True
yvar1 = 'stay2'
yvar2 = 'stay3'
for i in range(4):
    df = mergedfs[i].copy() #.loc[all_dfs[i]['lottery_year']==2023]
    df['stay1'] = 1 - df['change_company1']
    df['stay2'] = 1 - df['change_company2']
    df['stay3'] = 1 - df['change_company3']

    df_raw = mergedfs_clean[i].copy()
    df_raw['stay1'] = 1 - df_raw['change_company1']
    df_raw['stay2'] = 1 - df_raw['change_company2']
    df_raw['stay3'] = 1 - df_raw['change_company3']

    # keep only relevant columns + preserve panel index
    keep = ["stay1", "stay2", "stay3", "work_1yr", 'work_2yr', "winner", "ade", "weight_norm",'firm_year_fe', 'foia_indiv_id']
    df_raw = df_raw.reset_index()[keep].copy().set_index(['firm_year_fe', 'foia_indiv_id'])
    
    # wls ()
    if wls:
        m1s = m1s + [PanelOLS.from_formula(f'{yvar1} ~ winner + ade + EntityEffects', data=df_raw, weights=df_raw['weight_norm']).fit(cov_type = 'clustered')]

        m2s = m2s + [PanelOLS.from_formula(f'{yvar2} ~ winner + ade + EntityEffects', data=df_raw, weights=df_raw['weight_norm']).fit(cov_type = 'clustered')]
     
    # # DV mean for winners
    # print(df.loc[df['winner'] == 1]['stay1'].mean())
    # print(df.loc[df['winner'] == 1]['stay2'].mean())

    else:
        # weighted means (ols)
        m1s = m1s + [PanelOLS.from_formula(f'{yvar1} ~ winner + ade +EntityEffects', data=df).fit(cov_type = 'clustered')]

        m2s = m2s + [PanelOLS.from_formula(f'{yvar2} ~ winner + ade + EntityEffects', data=df).fit(cov_type = 'clustered')]

## outputting regression results
panelols_to_latex(m1s + m2s, [f'c = {c} (t{t})' for t in [1,2] for c in [2,4,6, 'inf'] ], verbose = True)

df = mergedfs_clean[3]
yvar_list = ['in_us1', 'in_us2', 'in_home_country1','in_home_country2', 'new_educ1', 'new_educ2']
x=panelols_to_latex([PanelOLS.from_formula(f'{yvar} ~ winner + ade + EntityEffects', data = df, weights = df['weight_norm']).fit(cov_type = 'clustered') for yvar in yvar_list], yvar_list, verbose = True)

df2 = mergedfs_clean[3]
x = panelols_to_latex([PanelOLS.from_formula(f'work_1yr ~ winner + ade + EntityEffects', data = d, weights = d['weight_norm']).fit(cov_type = 'clustered') for d in [df2, df2.loc[df2['high_rep_emp_ind']==1], df2.loc[df2['no_rep_emp_ind']==1], df2.loc[(round(df2['graddiff_agg']) < 3) & (df2['graddiff_agg'] >= -1)], df2.loc[(round(df2['graddiff_agg']) > 3) & (df2['graddiff_agg'] <= 6)]]] + [PanelOLS.from_formula(f'work_1yr ~ winner*graddiff3 + ade + EntityEffects', data = df2.assign(graddiff3 = np.where(round(df2['graddiff_agg']) == 3, 1, 0)), weights = df2['weight_norm']).fit(cov_type = 'clustered')], ['all (mult 2)', 'high rep employers only', 'no rep employers only', 'grad diff <3 yrs', 'grad diff >3 years', 'grad diff 3 yrs'], verbose = True)

df2 = mergedfs[3]
x = panelols_to_latex([PanelOLS.from_formula(f'work_1yr ~ winner + ade + EntityEffects', data = d).fit(cov_type = 'clustered') for d in [df2, df2.loc[df2['high_rep_emp_ind']==1], df2.loc[df2['no_rep_emp_ind']==1], df2.loc[(round(df2['graddiff']) == 3)]]] + [PanelOLS.from_formula(f'work_1yr ~ winner*graddiff3 + ade + EntityEffects', data = df2.assign(graddiff3 = np.where(round(df2['graddiff']) == 3, 1, 0))).fit(cov_type = 'clustered')], ['all (mult 2)', 'high rep employers only', 'no rep employers only', 'grad diff 3 yrs', 'interact'], verbose = True)

# print(compare({'c = 2': m1s[0], 'c = 2 (t2)': m2s[0], 'c = 4': m1s[1], 'c = 4 (t2)': m2s[1], 'c = 6': m1s[2], 'c = 6 (t2)': m2s[2]}, stars = True, precision = 'std_errors'))

# print(compare({'1. Baseline': m0_0s[0],'12. Baseline': m0_1s[0], '2. Pre-Filtered': m0_0s[1], '22. Pre-Filtered': m0_1s[1], '3. Post-Filtered': m0_0s[2],'33. Post-Filtered': m0_1s[2]}, stars = True, precision = 'std_errors'))

# print(compare({'1. Baseline': m2s[0],'12. Baseline': m3s[0], '2. Pre-Filtered': m2s[1], '22. Pre-Filtered': m3s[1], '3. Post-Filtered': m2s[2],'33. Post-Filtered': m3s[2]}, stars = True, precision = 'std_errors'))


print(compare([PanelOLS.from_formula(f'{yvar} ~ winner + ade + const + EntityEffects', data = mergedfs[1]).fit(cov_type = 'clustered') for yvar in ['in_us1', 'in_us2', 'promote1', 'promote2', 'new_educ1', 'new_educ2']], stars = True, precision = 'std_errors'))

for var in ['in_us1', 'in_us2', 'promote1', 'promote2', 'new_educ1', 'new_educ2']:
    print(mergedfs[1][var].mean())


#####################
# HISTOGRAMS
#####################
# sns.displot(df, x = 'enddatediff', bins = 12, hue = 'winner', col = 'year', alpha = 0.5, stat = 'density', multiple = 'dodge', common_norm = False)

# sns.histplot(merge_filt.loc[merge_filt['winner']], x = 'enddatediff',binwidth = 12, alpha = 0.5, stat = 'density')
# sns.histplot(merge_filt.loc[merge_filt['winner'] == 0], x = 'enddatediff',binwidth = 12, alpha = 0.5, stat = 'density')

# # dist of ade
# sns.histplot(df, x = 'ade')

# raw plots
plotdf = mergedfs[3].assign(graddiff = round(mergedfs[3]['graddiff'])).groupby(['winner', 'graddiff'])[['work_1yr','work_2yr','in_us1','in_us2','change_company1','change_company2']].mean().reset_index()

sns.regplot(data = mergedfs[3].loc[(mergedfs[3]['winner']==1) & (mergedfs[3]['graddiff']<5) & (mergedfs[3]['graddiff']>=-1)], x = 'graddiff', y = 'work_1yr', x_bins = 10)
sns.regplot(data = mergedfs[3].loc[(mergedfs[3]['winner']==0) & (mergedfs[3]['graddiff']<5) & (mergedfs[3]['graddiff']>=-1)], x = 'graddiff', y = 'work_1yr', x_bins = 10)


sns.scatterplot(data = plotdf.loc[(plotdf['graddiff']<5) & (plotdf['graddiff']>=-1)], x = 'graddiff', y = 'work_1yr', hue = 'winner')

sns.scatterplot(data = plotdf.loc[(plotdf['graddiff']<5) & (plotdf['graddiff']>=-1)], x = 'graddiff', y = 'work_2yr', hue = 'winner')