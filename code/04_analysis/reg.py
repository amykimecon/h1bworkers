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
rev_educ_clean = con.sql("SELECT *, ed_startdate AS startdate, ed_enddate AS enddate FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id, education_number ORDER BY matchscore DESC) AS match_order FROM rev_educ WHERE degree_clean != 'Non-Degree') WHERE match_order = 1")

# Importing User x Position-level Data (all positions, cleaned and deduplicated)
merged_pos = con.read_parquet(f'{root}/data/int/merged_pos_clean_aug8.parquet')

# removing duplicates, setting alt enddate as enddate if missing
merged_pos_clean = con.sql("SELECT * EXCLUDE (enddate), CASE WHEN alt_enddate IS NULL THEN enddate ELSE alt_enddate END AS enddate FROM merged_pos WHERE pos_dup_ind IS NULL OR pos_dup_ind = 0")

#all_merge_filts = [merge_filt_base, merge_filt_prefilt, merge_filt_postfilt]


#mergedf = merge_df(with_t_vars = True, postfilt = 'indiv', MATCH_MULT_CUTOFF = 2)
#####################
# CLEANING DATA
#####################
def merge_filt_clean(merge_filt, index = 'firm + year'):
    ## DEFINING VARIABLES
    # indicator for winning lottery
    merge_filt['winner'] = merge_filt['status_type'] == "SELECTED"

    # indicator for valid ADE
    merge_filt['ade'] = np.where((merge_filt['ade_ind'].isnull() == 0) & (merge_filt['ade_ind'] == 1), np.where((merge_filt['ade_year'].isnull()) | (merge_filt['ade_year'] <= merge_filt['lottery_year'].astype('int') - 1), 1, 0), 0)

    # indicator for still working at firm n years later
    merge_filt['work_1yr'] = np.where(merge_filt['enddatediff'] >= 12, 1, 0)
    merge_filt['work_2yr'] = np.where(merge_filt['enddatediff'] >= 24, 1, 0)
    merge_filt['work_3yr'] = np.where(merge_filt['lottery_year'] == '2024', np.nan, np.where(merge_filt['enddatediff'] >= 36, 1, 0)) 

    # promotion
    merge_filt['promote1'] = np.where((merge_filt['change_position1'] == 1) & (merge_filt['change_company1'] == 0), 1, 0)
    merge_filt['promote2'] = np.where((merge_filt['change_position2'] == 1) & (merge_filt['change_company1'] == 0), 1, 0)

    ## COLLAPSING TO APPLICATION LEVEL
    outcomes = ['change_company1', 'change_company2', 'promote1', 'promote2', 'in_us1', 'in_us2', 'ade', 'work_1yr', 'work_2yr', 'new_educ1', 'new_educ2', 'agg_compensation1', 'agg_compensation2']
    df = merge_filt.groupby(['foia_indiv_id', 'FEIN', 'female_ind', 'yob', 'lottery_year', 'foia_country', 'winner', 'n_apps', 'n_unique_country']).agg(
        **{var: (var, lambda x, var = var: np.average(x, weights = merge_filt.loc[x.index, 'weight_norm'])) for var in outcomes}
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

# all_dfs = [merge_filt_clean(m, 'firm + year') for m in all_merge_filts]
# all_df2s = [merge_filt_clean(m, 'firm x year') for m in all_merge_filts]

mergedfs_raw = [pd.read_parquet(f'{root}/data/int/merge_filt_mult{c}_aug13.parquet') for c in [2,4,6]]
mergedf_prefilt_raw = pd.read_parquet(f'{root}/data/int/merge_filt_prefilt_aug13.parquet')

mergedfs = [merge_filt_clean(df, 'firm x year') for df in mergedfs_raw]
#####################
# BALANCE TESTS
#####################
# avg max match weight by win
for mergedf in mergedfs_raw:
    mergedf['winner'] = mergedf['status_type'] == "SELECTED"
    print(mergedf.groupby(['foia_indiv_id','winner'])['weight_norm'].agg('max').reset_index().groupby('winner')['weight_norm'].agg('mean'))

# foia_dfs = [foia_raw.assign(dfname = 'All FOIA Apps'), foia_indiv.assign(dfname = 'Apps in Samp'), merge_filt_clean(merge_filt_base, '').assign(dfname = 'Matched Apps'), merge_filt_clean(merge_filt_prefilt, '').assign(dfname = 'Matched Excluding India/China+'), merge_filt_clean(merge_filt_postfilt, '').assign(dfname = 'Matched Post-Filtered')]
foia_dfs = [foia_indiv.assign(dfname = '0. Apps in Samp').assign(winner = np.where(foia_indiv['status_type'] == 'SELECTED', 1, 0)), all_dfs[0].assign(dfname = '1. Baseline'), all_dfs[1].assign(dfname = '2. Pre-Filtered'), all_dfs[2].assign(dfname = '3. Post-Filtered')]

foia_dfs_concat = pd.concat(foia_dfs)
foia_dfs_concat['india'] = np.where(foia_dfs_concat['foia_country'] == 'India', 1, 0)
foia_dfs_concat['china'] = np.where(foia_dfs_concat['foia_country'] == 'China', 1, 0)
foia_dfs_concat['age'] = foia_dfs_concat['lottery_year'].astype('int') - foia_dfs_concat['yob'].astype('int')

print(foia_dfs_concat.groupby('dfname')[['winner','female_ind', 'age', 'india', 'china', 'ade']].agg(lambda x: round(x.mean(), 2)).join(foia_dfs_concat.groupby(['dfname','FEIN', 'lottery_year'])[['n_apps','n_unique_country']].agg('mean').reset_index().groupby('dfname')[['n_apps','n_unique_country']].agg(lambda x: round(x.mean(), 2))).join(foia_dfs_concat.groupby('dfname').size().rename('n')).T)

#####################
# REGRESSIONS
#####################
yvar = 'change_company1'

m0_0s = []
m0_1s = []
m1s = []
m2s = []
m3s = []

for i in range(3):
    df = mergedfs[i] #.loc[all_dfs[i]['lottery_year']==2023]
    df['stay1'] = 1 - df['change_company1']
    df['stay2'] = 1 - df['change_company2']
    #df2 = all_df2s[i] #.loc[all_df2s[i]['lottery_year']==2023]

    # # ADE winner
    # m0_0s = m0_0s + [PanelOLS.from_formula(f'winner ~ const + ade + TimeEffects', data = df1).fit(cov_type = 'clustered')]

    # m0_1s = m0_1s + [PanelOLS.from_formula(f'winner ~ const + ade + EntityEffects', data = df2).fit(cov_type = 'clustered')]

    # # no ctrls
    # m1s = m1s + [PanelOLS.from_formula(f'{yvar} ~ const + winner', data = df1).fit(cov_type = 'robust')]

    # # year fes + ADE
    # m2s = m2s + [PanelOLS.from_formula(f'{yvar} ~ const + ade + winner + TimeEffects', data = df1).fit(cov_type = 'clustered')]

    # interacted fes + ADE
    # 
    m1s = m1s + [PanelOLS.from_formula(f'stay1 ~ winner + ade + const + EntityEffects', data=df).fit(cov_type = 'clustered')]

    m2s = m2s + [PanelOLS.from_formula(f'stay2 ~ winner + ade + const + EntityEffects', data=df).fit(cov_type = 'clustered')]

## outputting regression results
print(compare({'c = 2': m1s[0], 'c = 2 (t2)': m2s[0], 'c = 4': m1s[1], 'c = 4 (t2)': m2s[1], 'c = 6': m1s[2], 'c = 6 (t2)': m2s[2]}, stars = True, precision = 'std_errors'))

# print(compare({'1. Baseline': m0_0s[0],'12. Baseline': m0_1s[0], '2. Pre-Filtered': m0_0s[1], '22. Pre-Filtered': m0_1s[1], '3. Post-Filtered': m0_0s[2],'33. Post-Filtered': m0_1s[2]}, stars = True, precision = 'std_errors'))

print(compare({'1. Baseline': m2s[0],'12. Baseline': m3s[0], '2. Pre-Filtered': m2s[1], '22. Pre-Filtered': m3s[1], '3. Post-Filtered': m2s[2],'33. Post-Filtered': m3s[2]}, stars = True, precision = 'std_errors'))


print(compare([PanelOLS.from_formula(f'{yvar} ~ winner + ade + const + EntityEffects', data = mergedfs[1]).fit(cov_type = 'clustered') for yvar in ['in_us1', 'in_us2', 'promote1', 'promote2', 'new_educ1', 'new_educ2']], stars = True, precision = 'std_errors'))
# #####################
# # HISTOGRAMS
# #####################
# sns.displot(df, x = 'enddatediff', bins = 12, hue = 'winner', col = 'year', alpha = 0.5, stat = 'density', multiple = 'dodge', common_norm = False)

# sns.histplot(merge_filt.loc[merge_filt['winner']], x = 'enddatediff',binwidth = 12, alpha = 0.5, stat = 'density')
# sns.histplot(merge_filt.loc[merge_filt['winner'] == 0], x = 'enddatediff',binwidth = 12, alpha = 0.5, stat = 'density')

# # dist of ade
# sns.histplot(df, x = 'ade')