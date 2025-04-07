# File Description: Creating k_1500 to DOT code crosswalk 
# Author: Amy Kim
# Date Created: Mon Mar 31


# Imports and Paths
import duckdb as ddb
import pandas as pd
import wrds

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()
db = wrds.Connection()

#####################
# IMPORTING DATA
#####################
## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")
foia_dot_codes = con.sql("SELECT COUNT(*) AS n, DOT_CODE FROM foia_raw_file WHERE DOT_CODE != 'NA' GROUP BY DOT_CODE").df()
foia_dot_codes['share'] = foia_dot_codes['n']/foia_dot_codes.sum()['n']
foia_dot_codes['rank'] = foia_dot_codes['n'].rank(ascending = False)

## dot to onet code crosswalk
dot_onet_cw = pd.read_excel(f"{root}/data/crosswalks/DOT_to_ONET_SOC.xlsx", skiprows = 3, names = ['dot_code', 'dot_title', 'onet_code', 'onet_title'])

## revelio occ crosswalk
rev_occ_cw = db.raw_sql("SELECT * FROM revelio.individual_role_lookup")

#####################
# MERGING
#####################
dot_onet_cw['dot3_code'] = dot_onet_cw['dot_code'].apply(lambda x: x[0:3])

occ_cw = rev_occ_cw.merge(dot_onet_cw.groupby(['dot3_code','onet_code'])['onet_title'].agg('count').reset_index(), how = "left", on = "onet_code").merge(foia_dot_codes, how = 'left', left_on = 'dot3_code', right_on = 'DOT_CODE')

occ_cw[['n_foia','share_foia']] = occ_cw[['n','share']].fillna(0)
occ_cw['rank_foia'] = occ_cw['rank'].fillna(1000)

occ_cw_grouped = occ_cw.groupby(['role_k1500', 'role_k300', 'job_category', 'onet_code', 'onet_title_x']).agg(mean_n_foia = pd.NamedAgg(column = 'n_foia', aggfunc = 'mean'), max_share_foia = pd.NamedAgg(column = "share_foia", aggfunc = 'max'), min_rank = pd.NamedAgg(column = 'rank_foia', aggfunc = 'min')).reset_index().sort_values('min_rank')
occ_cw_grouped['top3occ'] = occ_cw_grouped['min_rank'] <= 3
occ_cw_grouped['top10occ'] = occ_cw_grouped['min_rank'] <= 10
occ_cw_grouped['mean_n_100'] = occ_cw_grouped['mean_n_foia'] >= 100

occ_cw_grouped.to_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")