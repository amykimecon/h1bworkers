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
from config import * # File Description: Merging H-1B and Revelio Individual Data
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

## revelio education data
rev_educ = con.read_parquet(f'{root}/data/int/rev_educ_long_aug8.parquet')

# collapsing to user x institution (for now use country with top score)
rev_educ_clean = con.sql("SELECT *, ed_startdate AS startdate, ed_enddate AS enddate FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id, education_number ORDER BY matchscore DESC) AS match_order FROM rev_educ WHERE degree_clean != 'Non-Degree') WHERE match_order = 1")

# Importing User x Position-level Data (all positions, cleaned and deduplicated)
merged_pos = con.read_parquet(f'{root}/data/int/merged_pos_clean_aug8.parquet')

# removing duplicates, setting alt enddate as enddate if missing
merged_pos_clean = con.sql("SELECT * EXCLUDE (enddate), CASE WHEN alt_enddate IS NULL THEN enddate ELSE alt_enddate END AS enddate FROM merged_pos WHERE pos_dup_ind IS NULL OR pos_dup_ind = 0")

#####################
# WRAPPER FUNCTIONS FOR MERGE
#####################
# GET DF 
def merge_df(con = con, rev_tab = 'rev_indiv', foia_tab = 'foia_indiv', with_t_vars = False, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, foia_prefilt = '', subregion = True, COUNTRY_SCORE_CUTOFF = 0, MONTH_BUFFER = 6, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8):
    return con.sql(merge(rev_tab, foia_tab, with_t_vars, postfilt, MATCH_MULT_CUTOFF, REV_MULT_COEFF, foia_prefilt, subregion, COUNTRY_SCORE_CUTOFF, MONTH_BUFFER, YOB_BUFFER, F_PROB_BUFFER)).df()

# WRAPPER
def merge(rev_tab = 'rev_indiv', foia_tab = 'foia_indiv', with_t_vars = False, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, foia_prefilt = '', subregion = True, COUNTRY_SCORE_CUTOFF = 0, MONTH_BUFFER = 6, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8):
    str_out = merge_filt_func(f"({merge_raw_func(rev_tab, foia_tab, foia_prefilt=foia_prefilt, subregion=subregion)})", postfilt=postfilt, MATCH_MULT_CUTOFF=MATCH_MULT_CUTOFF, REV_MULT_COEFF=REV_MULT_COEFF, COUNTRY_SCORE_CUTOFF=COUNTRY_SCORE_CUTOFF, MONTH_BUFFER=MONTH_BUFFER, YOB_BUFFER=YOB_BUFFER, F_PROB_BUFFER=F_PROB_BUFFER)

    if with_t_vars:
        str_out = f"SELECT * EXCLUDE (b.foia_indiv_id, b.user_id) FROM ({str_out}) AS a LEFT JOIN ({get_rel_year_inds_wide(f"({str_out})")}) AS b ON a.foia_indiv_id = b.foia_indiv_id AND a.user_id = b.user_id"

    return str_out

#####################
# REV X FOIA MERGE FUNCTIONS
#####################
# MERGE 
def merge_raw_func(rev_tab, foia_tab, foia_prefilt = '', subregion = True):
    if subregion:
        mergekey = 'subregion'
    else:
        mergekey = 'country'
    str_out = f"""SELECT *, a.country AS foia_country, b.country AS rev_country, COUNT(*) OVER(PARTITION BY foia_indiv_id) AS n_match_raw,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, first_startdate) AS startdatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, last_enddate) AS enddatediff, 
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, min_startdate_us::DATETIME) AS months_work_in_us,
            SUBSTRING(min_startdate, 1, 4)::INTEGER - 16 AS max_yob,
            DATEDIFF('month', ((lottery_year::INT - 1)::VARCHAR || '-03-01')::DATETIME, updated_dt::DATETIME) AS updatediff,
            (f_prob + f_prob_nt)/2 AS f_prob_avg,
            1 - ABS(female_ind - (f_prob + f_prob_nt)/2) AS f_score,
            CASE WHEN a.country = b.country THEN total_score ELSE 0 END AS country_score,
            (nanat_subregion_score + nt_subregion_score)/2 AS subregion_score,
            a.highest_ed_level AS foia_highest_ed_level, b.highest_ed_level AS rev_highest_ed_level
        FROM (SELECT * FROM {foia_tab} {foia_prefilt}) AS a LEFT JOIN {rev_tab} AS b ON a.rcid = b.rcid AND a.{mergekey} = b.{mergekey}"""
    return str_out

# FILTERS
# postfilt can be 'indiv' (filter on n_match_filt (# matches per app), then post-filter), 'emp' (filter on everything at employer level including match_mult_emp (avg # matches per app at employer level)), or anything else (no post-filtering)

def merge_filt_func(merge_raw_tab, postfilt = 'none', MATCH_MULT_CUTOFF = 4, REV_MULT_COEFF = 1, COUNTRY_SCORE_CUTOFF = 0, MONTH_BUFFER = 6, YOB_BUFFER = 5, F_PROB_BUFFER = 0.8):

    # warnings
    if MONTH_BUFFER >= 9:
        print('Warning! Month buffer of 9+ months will include positions started in the calendar year after the lottery, which may result in issues downstream')

    filt = f"""
    SELECT *, COUNT(*) OVER(PARTITION BY foia_indiv_id) AS n_match_filt FROM (
        SELECT *, 
        ROW_NUMBER() OVER(PARTITION BY foia_indiv_id, user_id ORDER BY country_score DESC, total_score DESC) AS match_order_ind,
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
    SELECT *, total_score/(SUM(total_score) OVER(PARTITION BY foia_indiv_id)) AS weight_norm FROM (
        SELECT foia_indiv_id, FEIN, lottery_year, rcid, user_id, fullname, foia_country, rev_country, subregion, country_score, subregion_score, female_ind, f_prob_avg, f_score, yob, est_yob, max_yob, n_match_raw, startdatediff, enddatediff, updatediff, stem_ind, foia_occ_ind, n_unique_country, min_h1b_occ_rank, months_work_in_us, n_apps, status_type, ade_ind, ade_year, foia_highest_ed_level, rev_highest_ed_level, prev_visa, field_clean, fields, positions, rcids, DOT_CODE, JOB_TITLE, n_match_filt,
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
    )"""

    if postfilt == 'indiv':
        str_out = f"SELECT * FROM ({str_out}) WHERE share_apps_matched_emp = 1 AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1"

    elif postfilt == 'emp':
        str_out = f"SELECT * FROM ({str_out}) WHERE share_apps_matched_emp = 1 AND rev_mult_emp < {REV_MULT_COEFF}*n_apps_matched_emp  AND n_unique_wintype_emp > 1 AND match_mult_emp <= {MATCH_MULT_CUTOFF}"

    return str_out



#####################
# LONG POSITION/EDUC MERGE
#####################
def get_rel_year_inds_wide(merge_tab, t0 = -1, t1 = 2):

    # join position and education data, unpivot long on variable, then pivot wide on variable x t
    str_out = f"""
    PIVOT 
        (UNPIVOT     
            (SELECT * EXCLUDE(b.foia_indiv_id, b.user_id, b.t) 
            FROM ({get_rel_year_inds_pos(merge_tab)}) AS a JOIN ({get_rel_year_inds_educ(merge_tab)}) AS b ON a.foia_indiv_id = b.foia_indiv_id AND a.user_id = b.user_id AND a.t = b.t)
        ON * EXCLUDE(foia_indiv_id, user_id, t) 
        INTO NAME var VALUE val)
    ON var || t USING FIRST(val)
    """

    return str_out

def get_rel_year_inds_educ(merge_tab, educ_tab = 'rev_educ_clean', t0 = -1, t1 = 5):
    """ Takes output of merge function and a table with user_id x education data and returns table at merge x t level with relevant variables (where t is relative to lottery year)

    Parameters
    -----------
    merge_tab : str
        Name or SQL string representing table that is output of merge functions
    educ_tab: str
        Name or SQL string representing table with user_id x education data
    t0, t1: optional inputs to help.long_by_year

    Returns
    -------
    String representing SQL query for table at merge x t level with relevant education variables
    """

    educlong = f"""
        SELECT *,
            -- indicator for education being in US
                CASE WHEN match_country = 'United States' THEN 1 ELSE 0 END AS educ_in_us,
            -- indicator for education being in country of birth
                CASE WHEN match_country = foia_country THEN 1 ELSE 0 END AS educ_in_home_country,
            -- indicator for masters
                CASE WHEN degree_clean = 'Master' OR degree_clean = 'MBA' THEN 1 ELSE 0 END AS masters,
            -- indicator for doctors
                CASE WHEN degree_clean = 'Doctor' THEN 1 ELSE 0 END AS doctors,
            -- indicator for education having been started before reference year
                CASE WHEN (MIN(t) OVER(PARTITION BY foia_indiv_id, user_id, education_number)) < 0 THEN 1 ELSE 0 END AS start_before,
            -- total number of educations in t
                COUNT(DISTINCT education_number) OVER(PARTITION BY foia_indiv_id, user_id, t) AS n_educ_t
        FROM ({get_long_by_year(merge_tab, educ_tab, long_tab_vars = ', degree_clean, match_country, startdate, enddate, education_number', t0 = t0, t1 = t1, enddatenull = "(CASE WHEN degree_clean = 'Master' OR degree_clean = 'MBA' THEN SUBSTRING(startdate, 1, 4)::INT + 2 ELSE SUBSTRING(startdate, 1, 4)::INT + 4 END)")})   
        ORDER BY foia_indiv_id, user_id, t 
    """

    educgroup = f""" 
    SELECT 
        foia_indiv_id, user_id, t,
        CASE WHEN COUNT(DISTINCT education_number) = 0 THEN 1 ELSE 0 END AS no_educations,
        MAX(educ_in_us) AS educ_in_us, MAX(educ_in_home_country) AS educ_in_home_country,
        MAX(masters) AS masters, MAX(doctors) AS doctors,
        MAX(CASE WHEN start_before = 0 AND educ_in_us = 1 THEN 1 ELSE 0 END) AS new_educ_in_us,
        MAX(CASE WHEN start_before = 0 AND educ_in_home_country = 1 THEN 1 ELSE 0 END) AS new_educ_in_home_country, 
        MAX(CASE WHEN start_before = 0 AND masters = 1 THEN 1 ELSE 0 END) AS new_masters, 
        MAX(CASE WHEN start_before = 0 AND doctors = 1 THEN 1 ELSE 0 END) AS new_doctors,
        MAX(CASE WHEN start_before = 0 THEN 1 ELSE 0 END) AS new_educ
    FROM ({educlong}) WHERE t IS NOT NULL
    GROUP BY foia_indiv_id, user_id, t"""

    return educgroup


def get_rel_year_inds_pos(merge_tab, pos_tab = 'merged_pos_clean', t0 = -1, t1 = 5):
    """ Takes output of merge function and a table with user_id x position data and returns table at merge x t level with relevant variables (where t is relative to lottery year)

    Parameters
    -----------
    merge_tab : str
        Name or SQL string representing table that is output of merge functions
    pos_tab: str
        Name or SQL string representing table with user_id x position data
    t0, t1: optional inputs to help.long_by_year

    Returns
    -------
    String representing SQL query for table at merge x t level with relevant position variables
    """

    # merging long on position x t
    poslong = f"""
        SELECT *, 
        -- indicator for position being in US
            CASE WHEN country = 'United States' THEN 1 ELSE 0 END AS in_us,
        -- indicator for position being in country of birth
            CASE WHEN country = foia_country THEN 1 ELSE 0 END AS in_home_country,
        -- indicator for being at same company as matched on in lottery
            CASE WHEN rcid = ref_rcid THEN 1 ELSE 0 END AS same_company,
        -- creating reference position number variable (first when all positions with t <= 0 and rcid = ref rcid are ordered by t desc, position number asc)
            MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id) AS ref_position_number,
        -- indicator for still being at reference position
            CASE WHEN position_number = (MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id)) THEN 1 ELSE 0 END AS same_position,
        -- indicator for position being started before reference position
            CASE WHEN position_number < (MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id)) THEN 1 ELSE 0 END AS start_before,
        -- imputed total compensation of reference position
            CASE WHEN position_number = (MAX(CASE WHEN ref_pos_priority = 1 THEN position_number ELSE 0 END) OVER(PARTITION BY foia_indiv_id, user_id)) THEN total_compensation ELSE 0 END AS ref_comp,
        -- total number of positions in t
            COUNT(DISTINCT position_id) OVER(PARTITION BY foia_indiv_id, user_id, t) AS n_pos_t
        FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY foia_indiv_id, user_id ORDER BY (CASE WHEN t <= 0 THEN 1 ELSE 0 END) DESC, t DESC, position_number) AS ref_pos_priority FROM ({
            get_long_by_year(merge_tab, pos_tab, long_tab_vars = ', position_id, position_number, b.rcid AS rcid, startdate, enddate, title_raw, company_raw, country, total_compensation', t0 = t0, t1 = t1)}))  
        ORDER BY foia_indiv_id, user_id, t 
    """

    # Filtering and grouping by t
    posgroup = f""" 
    SELECT 
        foia_indiv_id, user_id, t,
        CASE WHEN COUNT(DISTINCT position_number) = 0 THEN 1 ELSE 0 END AS no_positions,
        MAX(in_us) AS in_us, MAX(in_home_country) AS in_home_country,
        MAX(CASE WHEN start_before = 0 AND same_company = 0 THEN 1 ELSE 0 END) AS change_company,
        MAX(CASE WHEN start_before = 0 AND same_position = 0 THEN 1 ELSE 0 END) AS change_position,
        SUM(total_compensation * frac_t) AS agg_compensation,
        COUNT(*) AS n_pos,
        COUNT(CASE WHEN start_before = 0 THEN 1 END) AS n_pos_startafter,
        SUM(frac_t) AS frac_t
    FROM (
        SELECT *, COUNT(CASE WHEN start_before = 0 THEN 1 ELSE NULL END) OVER(PARTITION BY foia_indiv_id, user_id, t) AS n_start_after 
        FROM ({poslong})
        ) 
    WHERE t IS NOT NULL AND (start_before = 0 OR n_start_after = 0)
    GROUP BY foia_indiv_id, user_id, t"""

    return posgroup
    

def get_long_by_year(merge_tab, long_tab, long_tab_vars, t0 = -1, t1 = 5, enddatenull = '2025'):
    """ Takes output of merge function and a table long on user_id x event and returns SQL string for table long on merge x event x t where t is relative to lottery year 

    Parameters
    -----------
    merge_tab : str
        Name or SQL string representing table that is output of merge functions
    long_tab: str
        Name or SQL string representing table long on user_id x event where event has start and end date (e.g. position, education)
    long_tab_vars: str
        Additional vars from long_tab to keep for future steps (must start with comma)
    t0, t1, enddatenull: optional inputs to help.long_by_year

    Returns
    -------
    String representing SQL query for table long on merge x event x t
    """

    rawmerge = f"""
        SELECT foia_indiv_id, a.user_id, foia_country,
            lottery_year::INT - 1 AS ref_year, a.rcid AS ref_rcid {long_tab_vars}
        FROM {merge_tab} AS a LEFT JOIN {long_tab} AS b ON a.user_id = b.user_id
    """

    return help.long_by_year(tab = f'({rawmerge})', t0 = t0, t1 = t1, t_ref = 'x.ref_year', enddatenull = enddatenull, joinids = 'user_id, foia_indiv_id')


#mergetest = con.sql(merge(postfilt = 'indiv'))

#out = con.sql(get_rel_year_inds_wide('mergetest'))


# mergetest = con.sql(merge(postfilt = 'indiv'))

# out = con.sql(get_rel_year_inds_wide('mergetest'))


#####################
# DIFFERENT MERGE VERSIONS
# #####################
# con.sql(f"COPY ({merge(with_t_vars=True)}) TO '{root}/data/int/merge_filt_base_jul30.parquet'")
# con.sql(f"COPY ({merge(foia_prefilt = "WHERE subregion != 'Southern Asia' AND country != 'Canada' AND country != 'United Kingdom' AND country != 'Australia' AND country != 'China' AND country != 'Taiwan'")}) TO '{root}/data/int/merge_filt_prefilt_jul30.parquet'")
# con.sql(f"COPY ({merge(postfilt='indiv')}) TO '{root}/data/int/merge_filt_postfilt_jul30.parquet'")





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

