# File Description: Evaluating Merge Quality
# Author: Amy Kim
# Date Created: Jul 28 2025

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
import os
import sys

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
#####################
# CLEANING FOIA DATA
#####################
foiadf = con.sql(f"SELECT DISTINCT {','.join([col for col in foia_indiv.columns if col != 'rcid' and col != 'main_rcid'])} FROM foia_indiv").df().set_index(['foia_indiv_id'])

# indicator for winning lottery
foiadf['winner'] = foiadf['status_type'] == 'SELECTED'

# indicator for if winner previously on f1 visa
foiadf['fvisa'] = np.where(foiadf['prev_visa'].isnull() == 0, np.where(foiadf['prev_visa'] == 'F Visa', 1, 0), None)

# TODO: some indicators for top industries and occupations 

#####################
# MERGE SAMPLES
#####################


# function that takes in a list of mergedfs and a list of column names and outputs a modified foiadf with columns for the max, mean, and sum total scores of the individual id in the foiadf + indicator for if foia_indiv_id was successfully merged
def foia_compare_df(mergedfs, colnames, foiadf = foiadf):
    if len(mergedfs) != len(colnames):
        print("need to input colnames same size as mergedfs")

    foiadf_out = foiadf.copy()

    for i in range(len(mergedfs)):
        foiadf_out = foiadf_out.merge(mergedfs[i].groupby('foia_indiv_id')['total_score'].agg(**{
            f'{colnames[i]}_maxscore':'max', 
            f'{colnames[i]}_avgscore': 'mean',
            f'{colnames[i]}_sumscore': 'sum',
            f'{colnames[i]}_n': 'count',
            }), how = 'left', left_index = True, right_index = True, indicator = colnames[i])
        
        foiadf_out[colnames[i]] = np.where(foiadf_out[colnames[i]] == 'both', 1, 0)
        
    return foiadf_out 

# TODO: MODIFY SO THAT POSITION ONLY CONSIDERS MATCHES IF WITH RIGHT RCID
# function that takes in a mergedf and outputs some stats on how well it's matched on highest degree, field of study, job title
def merge_eval(mergedf):
    # cleaning
    merge_clean = mergedf.copy()

    # highest ed level
    merge_clean['highest_ed_ind'] = np.where((
        (merge_clean['foia_highest_ed_level'] == '')|(merge_clean['rev_highest_ed_level'] == '')|(merge_clean['foia_highest_ed_level'] == 'na')|(merge_clean['rev_highest_ed_level'] == 'na')|(pd.isnull(merge_clean['foia_highest_ed_level']))|(pd.isnull(merge_clean['rev_highest_ed_level']))), 0, 1)
    
    merge_clean['highest_ed_match'] = merge_clean.apply(lambda x: None if x['highest_ed_ind'] == 0 else x['weight_norm']*(x['foia_highest_ed_level'] == x['rev_highest_ed_level']), axis = 1)

    # field of study
    merge_clean = merge_listmatch(merge_clean, listcol = 'fields', matchcol = 'field_clean', prefix = 'field_')

    # position title
    merge_clean = merge_listmatch(merge_clean, listcol = 'positions', matchcol = 'job_title', prefix = 'job_')

    # all matching
    merge_clean['all_match'] = merge_clean.apply(lambda x: None if x['highest_ed_ind'] == 0 or x['field_ind'] == 0 or x['job_ind'] == 0 else x['weight_norm']*(x['highest_ed_match'] > 0 and x['field_match'] > 0 and x['job_match'] > 0), axis = 1)

    # any matching (field/job)
    merge_clean['any_match'] = merge_clean.apply(lambda x: None if x['field_ind'] == 0 and x['job_ind'] == 0 else x['weight_norm']*(x['field_match'] > 0 or x['job_match'] > 0), axis = 1)

    varnames = ['highest_ed_match', 'field_match', 'job_match', 'any_match']
    labels = ['highest ed level', 'field of study', 'job title', 'any']
    funcs = ['sum', 'max', 'count']

    # aggregating to foia_indiv_id level
    merge_agg = merge_clean.groupby('foia_indiv_id').agg(
        **{f'{var}_{func}': (var, func) for var in varnames for func in funcs}
    )
    

    merge_agg_counts = merge_clean.groupby('foia_indiv_id')[varnames].agg(lambda x: (x > 0).sum())

    print("Reporting match quality stats:")
    for i in range(len(varnames)):
        print(labels[i])
        n = merge_agg.loc[merge_agg[f"{varnames[i]}_count"] > 0].shape[0]
        avgmult = merge_clean.loc[pd.isnull(merge_clean[f'{varnames[i]}'])==0].shape[0]/n
        print(f"---{n} applications with valid info")
        print(f"---{merge_agg.loc[merge_agg[f"{varnames[i]}_max"] > 0].shape[0]} ({round(100*merge_agg.loc[merge_agg[f"{varnames[i]}_max"] > 0].shape[0]/n, 2)}%) have at least one match with the correct {labels[i]}")
        print(f"---of those with at least one correct match, the mean number of matches matching on {labels[i]} is {round(merge_agg_counts.loc[merge_agg_counts[varnames[i]] >= 1][varnames[i]].mean(),2)} (avg multiplicity among apps with valid info: {round(avgmult,2)})")



# function that explodes mergedf on column `listcol` of lists, evaluates match on `matchcol`, collapses back to match level
def merge_listmatch(mergedf, listcol, matchcol, prefix):
    merge_out_long = mergedf.reset_index()[['user_id','foia_indiv_id',listcol,matchcol]].explode(listcol)

    # skip if na, missing, blank
    merge_out_long[f'{prefix}ind'] = np.where((
        (merge_out_long[listcol] == '')|(merge_out_long[matchcol] == '')|(merge_out_long[listcol] == 'na')|(merge_out_long[matchcol] == 'na')|(pd.isnull(merge_out_long[listcol]))|(pd.isnull(merge_out_long[matchcol]))), 0, 1)
    
    # levenshtein normalized similarity
    merge_out_long[f'{prefix}levsim'] = merge_out_long.apply(lambda x: 0 if x[f'{prefix}ind'] == 0 else Levenshtein.normalized_similarity(x[matchcol], x[listcol]), axis=1)
    
    # check if one column substring of another
    merge_out_long[f'{prefix}substr'] = merge_out_long.apply(lambda x: 0 if x[f'{prefix}ind'] == 0 else x[matchcol] in x[listcol] or x[matchcol] in x[listcol], axis = 1)

    # aggregating to match level
    merge_out = mergedf.merge(merge_out_long.groupby(['user_id','foia_indiv_id'])[[f'{prefix}ind', f'{prefix}levsim', f'{prefix}substr']].agg('max'), how = 'left', on = ['user_id', 'foia_indiv_id'])

    merge_out[f'{prefix}match'] = merge_out.apply(lambda x: None if x[f'{prefix}ind'] == 0 else x['weight_norm']*(x[f'{prefix}levsim'] >= 0.85 or x[f'{prefix}substr'] == 1), axis = 1)

    return merge_out



# 2. highest degree

# 3. pivot long on major, check if match

# 4. pivot long on job title, check if match

# mergedf = merge_df().set_index(['foia_indiv_id'])
# foiadf_withmerge = foia_compare_df([mergedf], ['baseline'])
# merge_eval(mergedf)
# merge_eval(merge_df(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'"))

# merge_eval(merge_df(postfilt = 'indiv'))



#####################
# TEMP: ESTIMATING BENCHMARK PARAMETERS
#####################
#mergedfs = [merge_df(with_t_vars=True), merge_df(with_t_vars=True,foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'"), merge_df(with_t_vars=True,postfilt = 'indiv')]

mergedfs = [merge_df(with_t_vars=True, postfilt = 'indiv', MATCH_MULT_CUTOFF = c) for c in [2,4,6]]

def s3(p, g):
    return (1-p)**2/((1+g)**2 + (1+g)*(1-p) + (1-p)**2)

for mergedf in mergedfs:
    mergedf['promote1'] = np.where((mergedf['change_position1'] == 1) & (mergedf['change_company1'] == 0), 1, 0)
    mergedf['promote2'] = np.where((mergedf['change_position2'] == 1) & (mergedf['change_company1'] == 0), 1, 0)
    # estimating r
    for v in ['in_us1', 'in_us2', 'promote1', 'promote2', 'new_educ1', 'new_educ2']:
        print(v)
        mergedf[f'{v}alt'] = mergedf[v] * mergedf['weight_norm']
        print(mergedf.groupby('foia_indiv_id')[f'{v}alt'].agg('sum').mean())
    # mergedf['leave1norm'] = mergedf['weight_norm'] * mergedf['change_company1']
    # mergedf['leave2norm'] = mergedf['weight_norm'] * mergedf['change_company2']
    # print(1 - mergedf.groupby('foia_indiv_id')[['leave1norm', 'leave2norm']].agg('sum').mean())

    # print(mergedf.groupby('foia_indiv_id')[['in_us1', 'in_us2', 'promote1', 'promote2', 'new_educ1', 'new_educ2']].agg('sum').mean())

    # # estimating w 
    # print(f" Estimated w (method 1): {round(mergedf.groupby('foia_indiv_id')['weight_norm'].agg('max').mean(),4)}")

    # print(f" Estimated w (method 2): {round(mergedf.groupby('foia_indiv_id')['weight_norm'].agg('mean').mean(),4)}")

    # print(f"Multiplicity: {mergedf.shape[0]/mergedf.groupby('foia_indiv_id').size().shape[0]}")

a = mergedfs[0]

a.groupby('lottery')
# con.sql(f"COPY ({merge()}) TO '{root}/data/int/merge_filt_base_jul30.parquet'")
# con.sql(f"COPY ({merge(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'")}) TO '{root}/data/int/merge_filt_prefilt_jul30.parquet'")
# con.sql(f"COPY ({merge(postfilt='indiv')}) TO '{root}/data/int/merge_filt_postfilt_jul30.parquet'")

# merge1 = con.sql(f"SELECT foia_indiv_id, lottery_year, FEIN, status_type, MAX(total_score) AS max_score, MEAN(total_score) AS mean_score, SUM(total_score) AS sum_score, MAX() FROM ({merge()}) GROUP BY foia_indiv_id, FEIN, lottery_year, status_type")


# con.sql(f"COPY ({merge()}) TO '{root}/data/int/merge_filt_base_jul30.parquet'")
# con.sql(f"COPY ({merge(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'")}) TO '{root}/data/int/merge_filt_prefilt_jul30.parquet'")
# con.sql(f"COPY ({merge(postfilt='indiv')}) TO '{root}/data/int/merge_filt_postfilt_jul30.parquet'")


# print(con.sql('SELECT COUNT(*) AS n_match, COUNT(DISTINCT foia_indiv_id) AS n_app FROM merge1'))

# # baseline merge with restrictive country score cutoff
# merge2 = con.sql(merge(COUNTRY_SCORE_CUTOFF=0.3))
# print(con.sql('SELECT COUNT(*) AS n_match, COUNT(DISTINCT foia_indiv_id) AS n_app FROM merge2'))

# # pre-filt merge (less restrictive)
# merge3 = con.sql(merge(foia_prefilt = "WHERE country != 'China' AND country != 'India'"))
# print(con.sql('SELECT COUNT(*) AS n_match, COUNT(DISTINCT foia_indiv_id) AS n_app FROM merge3'))

# # pre-filt merge (more restrictive)
# merge4 = con.sql(merge(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'"))
# print(con.sql('SELECT COUNT(*) AS n_match, COUNT(DISTINCT foia_indiv_id) AS n_app FROM merge4'))

# # post-filt merge (employer level)
# merge5 = con.sql(merge(postfilt='emp'))
# print(con.sql('SELECT COUNT(*) AS n_match, COUNT(DISTINCT foia_indiv_id) AS n_app FROM merge5'))

# # post-filt merge (indiv level)
# merge6 = con.sql(merge(postfilt='indiv'))
# print(con.sql('SELECT COUNT(*) AS n_match, COUNT(DISTINCT foia_indiv_id) AS n_app FROM merge6'))

