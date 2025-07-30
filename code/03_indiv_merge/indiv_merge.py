# File Description: Merging H-1B and Revelio Individual Data
# Author: Amy Kim
# Date Created: Jun 30 2025

# Imports and Paths
import duckdb as ddb
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

con = ddb.connect()

#####################
# IMPORTING DATA
#####################
## foia
foia_indiv = con.read_parquet(f'{root}/data/clean/foia_indiv.parquet')

## revelio
rev_indiv = con.read_parquet(f'{root}/data/clean/rev_indiv.parquet')


#####################
# HELPER FUNCTIONS
#####################
# WRAPPER
def merge(rev_tab = 'rev_indiv', foia_tab = 'foia_indiv', postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, foia_prefilt = '', subregion = True, COUNTRY_SCORE_CUTOFF = 0, MONTH_BUFFER = 6, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8):
    str_out = merge_filt_func(f"({merge_raw_func(rev_tab, foia_tab, foia_prefilt=foia_prefilt, subregion=subregion)})", postfilt=postfilt, MATCH_MULT_CUTOFF=MATCH_MULT_CUTOFF, REV_MULT_COEFF=REV_MULT_COEFF, COUNTRY_SCORE_CUTOFF=COUNTRY_SCORE_CUTOFF, MONTH_BUFFER=MONTH_BUFFER, YOB_BUFFER=YOB_BUFFER, F_PROB_BUFFER=F_PROB_BUFFER)

    return str_out


# MERGE 
def merge_raw_func(rev_tab, foia_tab, foia_prefilt = '', subregion = True):
    if subregion:
        mergekey = 'subregion'
    else:
        mergekey = 'country'
    str_out = f"""SELECT *, a.country AS foia_country, b.country AS rev_country, COUNT(*) OVER(PARTITION BY foia_indiv_id, a.rcid) AS n_match_raw,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, first_startdate) AS startdatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, last_enddate) AS enddatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, min_startdate_us::DATETIME) AS months_work_in_us,
            SUBSTRING(min_startdate, 1, 4)::INTEGER - 16 AS max_yob,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, updated_dt::DATETIME) AS updatediff,
            (f_prob + f_prob_nt)/2 AS f_prob_avg,
            1 - ABS(female_ind - (f_prob + f_prob_nt)/2) AS f_score,
            CASE WHEN a.country = b.country THEN total_score ELSE 0 END AS country_score,
            (nanat_subregion_score + nt_subregion_score)/2 AS subregion_score
        FROM (SELECT * FROM {foia_tab} {foia_prefilt}) AS a LEFT JOIN {rev_tab} AS b ON a.rcid = b.rcid AND a.{mergekey} = b.{mergekey}"""
    return str_out

# FILTERS
# postfilt can be 'indiv' (filter on n_match_filt (# matches per app), then post-filter), 'emp' (filter on everything at employer level including match_mult_emp (avg # matches per app at employer level)), or anything else (no post-filtering)

def merge_filt_func(merge_raw_tab, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, COUNTRY_SCORE_CUTOFF = 0, MONTH_BUFFER = 6, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8):
    filt = f"""
    SELECT *, COUNT(*) OVER(PARTITION BY foia_indiv_id, rcid) AS n_match_filt FROM (
        SELECT *, 
        ROW_NUMBER() OVER(PARTITION BY foia_indiv_id, rcid, user_id ORDER BY country_score DESC, total_score DESC) AS match_order_ind,
        MAX(country_score) OVER(PARTITION BY foia_indiv_id) AS max_country_score,
        FROM {merge_raw_tab}
        WHERE f_score >= 1 - {F_PROB_BUFFER} AND 
            (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
            AND startdatediff <= {0+MONTH_BUFFER} AND startdatediff >= {-36 - MONTH_BUFFER} AND enddatediff >= {0-MONTH_BUFFER}
        ) 
    WHERE match_order_ind = 1 AND stem_ind = 1 AND foia_occ_ind = 1 AND (country_score > {COUNTRY_SCORE_CUTOFF} OR max_country_score <= {COUNTRY_SCORE_CUTOFF})
    """
    
    if postfilt == 'indiv':
        filt = f"SELECT * FROM ({filt}) WHERE n_match_filt <= {MATCH_MULT_CUTOFF}"
    
    str_out = f"""
        SELECT foia_indiv_id, FEIN, lottery_year, rcid, user_id, fullname, foia_country, rev_country, subregion, country_score, subregion_score, female_ind, f_prob_avg, f_score, yob, est_yob, max_yob, n_match_raw, startdatediff, enddatediff, updatediff, stem_ind, foia_occ_ind, n_unique_country, min_h1b_occ_rank, months_work_in_us, n_apps, status_type, ade_ind, ade_year,
            COUNT(*) OVER(PARTITION BY foia_indiv_id, rcid) AS n_match_filt,
            (COUNT(DISTINCT foia_indiv_id) OVER(PARTITION BY FEIN, lottery_year))/n_apps AS share_apps_matched_emp,
            COUNT(DISTINCT user_id) OVER(PARTITION BY FEIN, lottery_year) AS n_rev_users_emp,
            (COUNT(*) OVER(PARTITION BY FEIN, lottery_year))/(COUNT(DISTINCT foia_indiv_id) OVER(PARTITION BY FEIN, lottery_year)) AS match_mult_emp,
            (COUNT(*) OVER(PARTITION BY FEIN, lottery_year))/(COUNT(DISTINCT user_id) OVER(PARTITION BY FEIN, lottery_year)) AS rev_mult_emp,
            COUNT(DISTINCT foia_indiv_id) OVER(PARTITION BY FEIN, lottery_year) AS n_apps_matched_emp,
            COUNT(DISTINCT status_type) OVER(PARTITION BY FEIN, lottery_year) AS n_unique_wintype_emp,
        (CASE WHEN est_yob IS NULL THEN 0
            WHEN ABS(est_yob - yob::INTEGER) <= 1 THEN 1
            WHEN est_yob - yob::INTEGER <= 3 AND est_yob - yob::INTEGER >= 2 THEN 0.8
            WHEN est_yob - yob::INTEGER >= -3 AND est_yob - yob::INTEGER <= -2 THEN 0.6
            WHEN est_yob - yob::INTEGER <= {YOB_BUFFER} AND est_yob - yob::INTEGER >= 4 THEN 0.4
            WHEN est_yob - yob::INTEGER >= -{YOB_BUFFER} AND est_yob - yob::INTEGER <= -4 THEN 0.2
        END)/6 + 
        ((f_score - (1 - {F_PROB_BUFFER}))/{F_PROB_BUFFER})/6 +
        subregion_score/6 + country_score/2 AS total_score
        FROM ({filt})
        """

    if postfilt == 'indiv':
        str_out = f"SELECT * FROM ({str_out}) WHERE share_apps_matched_emp = 1 AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1"

    elif postfilt == 'emp':
        str_out = f"SELECT * FROM ({str_out}) WHERE share_apps_matched_emp = 1 AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1 AND match_mult_emp <= {MATCH_MULT_CUTOFF}"

    return str_out


#####################
# DIFFERENT MERGE VERSIONS
#####################
con.sql(f"COPY ({merge()}) TO '{root}/data/int/merge_filt_base_jul30.parquet'")
con.sql(f"COPY ({merge(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'")}) TO '{root}/data/int/merge_filt_prefilt_jul30.parquet'")
con.sql(f"COPY ({merge(postfilt='indiv')}) TO '{root}/data/int/merge_filt_postfilt_jul30.parquet'")





# con.sql(f"""CREATE OR REPLACE TABLE merge_raw AS {merge_raw_func('rev_indiv', 'foia_indiv')}""")

# con.sql(f"""CREATE OR REPLACE TABLE merge_raw_subregion AS {merge_raw_func('rev_indiv', 'foia_indiv', subregion=True)}""")

# con.sql(f"""CREATE OR REPLACE TABLE merge_raw_prefilt AS {merge_raw_func('rev_indiv', "(SELECT * FROM foia_indiv WHERE country != 'China' AND country != 'India' AND country != 'Taiwan' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'Nepal' AND country != 'Pakistan')")}""")


# con.sql(f"""CREATE OR REPLACE TABLE merge_raw_prefilt_subregion AS {merge_raw_func('rev_indiv', "(SELECT * FROM foia_indiv WHERE country != 'China' AND country != 'India' AND country != 'Taiwan' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'Nepal' AND country != 'Pakistan')", subregion = True)}""")

# merge_filt_base = con.sql(merge_filt_func('merge_raw'))
# merge_filt_subregion = con.sql(merge_filt_func('merge_raw_subregion'))

# merge_filt_prefilt = con.sql(merge_filt_func('merge_raw_prefilt'))

# merge_filt_prefilt_subregion = con.sql(merge_filt_func('merge_raw_prefilt_subregion'))

# MATCH_MULT_CUTOFF = 4
# REV_MULT_COEFF = 1
# # version 1: filtering on avg match_mult at the firm x year level
# merge_filt_postfilt = con.sql(f'SELECT * FROM ({merge_filt_func('merge_raw')}) WHERE share_apps_matched_emp = 1 AND match_mult_emp <= {MATCH_MULT_CUTOFF} AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1')
# #print(merge_filt_postfilt.shape)

# # version 2: filtering on match_mult at the foia app level, then filtering on firm stuff (more restrictive)
# merge_filt_postfilt2 = con.sql(f'SELECT * FROM (SELECT (COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year))/n_apps AS share_apps_matched_emp, (COUNT(*) OVER(PARTITION BY FEIN, lottery_year))/(COUNT(DISTINCT user_id) OVER(PARTITION BY FEIN, lottery_year)) AS rev_mult_emp, COUNT(DISTINCT foia_temp_id) OVER(PARTITION BY FEIN, lottery_year) AS n_apps_matched_emp,COUNT(DISTINCT status_type) OVER(PARTITION BY FEIN, lottery_year) AS n_unique_wintype_emp, * FROM ({merge_filt_func('merge_raw')}) WHERE n_match_filt <= {MATCH_MULT_CUTOFF}) WHERE share_apps_matched_emp = 1 AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1')
# #print(merge_filt_postfilt2.shape)


# con.sql(f"COPY foia_indiv TO '{root}/data/int/foia_merge_samp_jul23.parquet'")
# con.sql(f"COPY merge_filt_base TO '{root}/data/int/merge_filt_base_jul23.parquet'")
# con.sql(f"COPY merge_filt_postfilt2 TO '{root}/data/int/merge_filt_postfilt_jul23.parquet'")
# con.sql(f"COPY merge_filt_prefilt TO '{root}/data/int/merge_filt_prefilt_jul23.parquet'")

