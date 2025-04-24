# File Description: Initial Analysis based on R Merge
# Author: Amy Kim
# Date Created: Tue Mar 25

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
con.create_function("get_fiscal_year", ah.get_fiscal_year_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_fiscal_year_foia", ah.get_fiscal_year_foia_sql, ["VARCHAR"], "FLOAT")
    
con.create_function("get_quarter", ah.get_quarter_sql, ["VARCHAR"], "FLOAT")

con.create_function("get_quarter_foia", ah.get_quarter_foia_sql, ["VARCHAR"], "FLOAT")

#####################
# IMPORTING DATA
#####################
## duplicate rcids (companies that appear more than once in linkedin data)
dup_rcids = con.read_csv(f"{root}/data/int/dup_rcids_mar20.csv")
dup_rcids_unique = con.sql("SELECT rcid, main_rcid FROM dup_rcids GROUP BY rcid, main_rcid")

## matched company data from R
rmerge = con.read_csv(f"{root}/data/int/good_match_ids_mar20.csv")

## raw FOIA bloomberg data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")

## raw FOIA I129 data
foia_i129_file = con.read_csv(f"{root}/data/raw/foia_i129/i129_allfys.csv")

## joining raw FOIA data with merged data to get foia_ids in raw foia data
foia_with_ids = con.sql("SELECT ROW_NUMBER() OVER() AS bb_id, foia_id, BEN_COUNTRY_OF_BIRTH, BEN_SEX, JOB_TITLE, DOT_CODE, NAICS_CODE, S3Q1, PET_ZIP, WORKSITE_CITY, WORKSITE_STATE, BASIS_FOR_CLASSIFICATION, REQUESTED_ACTION, BEN_PFIELD_OF_STUDY, BEN_CURRENT_CLASS, BEN_EDUCATION_CODE, BEN_COMP_PAID, valid_from, valid_to, employer_name, i129_employer_name, NUM_OF_EMP_IN_US, a.FEIN AS FEIN, status_type, ben_multi_reg_ind, rec_date, first_decision_date, FIRST_DECISION, a.lottery_year AS lottery_year FROM ((SELECT * FROM foia_raw_file WHERE NOT FEIN = '(b)(3) (b)(6) (b)(7)(c)') AS a LEFT JOIN (SELECT lottery_year, FEIN, foia_id FROM rmerge GROUP BY lottery_year, FEIN, foia_id) AS b ON a.lottery_year = b.lottery_year AND a.FEIN = b.FEIN)")

## revelio data (pre-filtered to only companies in rmerge)
merged_pos = con.read_parquet(f"{root}/data/int/rev_merge_mar20.parquet")
occ_cw = con.read_csv(f"{root}/data/crosswalks/rev_occ_to_foia_freq.csv")

## joining position data to dup_rcids to get main_rcid and preprocessing (getting fiscal year/quarter, ordering by startq within user x company)
merged_pos_full = con.sql(
    f"""SELECT ROW_NUMBER() OVER(PARTITION BY main_rcid, user_id ORDER BY get_quarter(startdate)) AS rank, 
    COUNT(position_id) OVER(PARTITION BY main_rcid, user_id) AS n_pos,
    weight,
        get_quarter(startdate) AS startq,
        get_fiscal_year(startdate) AS startfy,
        CASE WHEN enddate IS NULL AND startdate IS NOT NULL THEN 2025.25 ELSE get_quarter(enddate) END AS endq,
        CASE WHEN enddate IS NULL AND startdate IS NOT NULL THEN 2025 ELSE get_fiscal_year(enddate) END AS endfy,
        country, main_rcid, user_id, 
        role_k1500, top3occ, top10occ, mean_n_100
    FROM (SELECT startdate, enddate, user_id, country, weight, pos.role_k1500 AS role_k1500, top3occ, top10occ, mean_n_100, position_id,
            CASE WHEN main_rcid IS NULL THEN pos.rcid ELSE main_rcid END AS main_rcid
        FROM ((SELECT * FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY position_id) AS n_rank FROM merged_pos) WHERE n_rank = 1) AS pos
        LEFT JOIN
            (SELECT role_k1500, top3occ, top10occ, mean_n_100 FROM occ_cw) AS occ_cw 
        ON pos.role_k1500 = occ_cw.role_k1500
        LEFT JOIN 
            dup_rcids_unique AS dup_cw
        ON pos.rcid = dup_cw.rcid))""")

#####################################
# 1. COLLAPSING DATA FOR ANALYSIS 
#       (each row is foia_id X fy = (FEIN x lottery_year x RCID) X fy)
#####################################
# STEP ONE: get central crosswalk of matched IDs unique at the main_rcid x FEIN x lottery_year level (note: RMERGE is unique at the FEIN x year level (no duplicate main_RCIDs), but is not unique at the main_rcid x year level (may have duplicate FEINs), so need to collapse)
id_merge = con.sql("""
-- collapse matched IDs to be unique at the rcid x FEIN x lottery_year level
SELECT lottery_year, main_rcid, foia_id
FROM rmerge 
GROUP BY foia_id, lottery_year, main_rcid
""")

# STEP TWO: clean and collapse FOIA raw data to foia_id level
foia_for_merge = con.sql("""
SELECT foia_id, FIRST(employer_name) AS company_FOIA, lottery_year,
    MAX(CASE WHEN NOT NUM_OF_EMP_IN_US = 'NA' THEN NUM_OF_EMP_IN_US::INTEGER END) AS n_us_employees,
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_apps, 
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 THEN 1 END) AS n_success,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 AND rec_date != 'NA' THEN 1 END) AS n_i129,
    COUNT(CASE WHEN BEN_CURRENT_CLASS = 'UU' THEN 1 END) AS n_uu
FROM foia_with_ids WHERE foia_id IS NOT NULL GROUP BY foia_id, lottery_year
""")

# STEP THREE POINT ONE: collapse revelio data to main_rcid x quarter level (get counts by quarter of start date)
rev_for_merge_startq = con.sql("""
SELECT main_rcid, startq AS t, FIRST(startfy) AS startfy,
    COUNT(*) AS new_positions,
    SUM(weight) AS new_positions_weighted,
    COUNT(CASE WHEN top3occ = 1 THEN 1 END) AS new_positions_top3,
    COUNT(CASE WHEN top10occ = 1 THEN 1 END) AS new_positions_top10,
    COUNT(CASE WHEN mean_n_100 = 1 THEN 1 END) AS new_positions_highn,
    COUNT(CASE WHEN rank = 1 THEN 1 END) AS new_hires,
    COUNT(CASE WHEN rank != 1 THEN 1 END) AS promotions,
    SUM(CASE WHEN rank = 1 THEN weight ELSE 0 END) AS new_hires_weighted,
    COUNT(CASE WHEN rank = 1 AND top3occ = 1 THEN 1 END) AS new_hires_top3,
    COUNT(CASE WHEN rank = 1 AND top10occ = 1 THEN 1 END) AS new_hires_top10,
    COUNT(CASE WHEN rank = 1 AND mean_n_100 = 1 THEN 1 END) AS new_hires_highn
FROM 
    merged_pos_full
WHERE country = 'United States'
GROUP BY main_rcid, startq
""")

# STEP THREE POINT TWO: join revelio data with full range of quarters to get main_rcid x quarter level employee counts
rev_for_merge_nq = con.sql(f"SELECT t, main_rcid, COUNT(*) AS n_emp, SUM(weight) AS n_emp_weighted, COUNT(CASE WHEN top3occ = 1 THEN 1 END) AS n_emp_top3, COUNT(CASE WHEN top10occ = 1 THEN 1 END) AS n_emp_top10, COUNT(CASE WHEN mean_n_100 = 1 THEN 1 END) AS n_emp_highn FROM (SELECT time.y AS t, main_rcid, user_id, weight, top3occ, top10occ, mean_n_100 FROM (SELECT generate_series/4 FROM generate_series(2000*4,2025*4)) AS time(y) LEFT OUTER JOIN merged_pos_full AS a ON time.y between startq AND endq GROUP BY time.y, main_rcid, user_id, weight, top3occ, top10occ, mean_n_100) GROUP BY t, main_rcid")

# STEP THREE POINT THREE: get counts by quarter of end date
rev_for_merge_endq = con.sql("""
SELECT main_rcid, endq AS t, FIRST(endfy) AS endfy,
    COUNT(*) AS leaves,
    COUNT(CASE WHEN rank = n_pos THEN 1 END) AS quits
FROM 
    merged_pos_full
WHERE country = 'United States'
GROUP BY main_rcid, endq
""")

# STEP THREE: join both revelio collapsed datasets onto cross join of foia_ids and t to code missings as 0s
rev_for_merge = con.sql(
"""
    SELECT * FROM(
        (SELECT * FROM (SELECT generate_series/4 AS t FROM generate_series(2000*4,2025*4)) CROSS JOIN (SELECT main_rcid FROM id_merge GROUP BY main_rcid)) AS a
        LEFT JOIN 
        rev_for_merge_startq AS b
        ON a.main_rcid = b.main_rcid AND a.t = b.t
        LEFT JOIN 
        rev_for_merge_nq AS c
        ON a.main_rcid = c.main_rcid AND a.t = c.t
        LEFT JOIN
        rev_for_merge_endq AS d
        ON a.main_rcid = d.main_rcid AND a.t = d.t
    )
""")

# STEP FOUR: merge all datasets
merged_for_analysis = con.sql(
"""
SELECT ids.lottery_year AS lottery_year, ids.main_rcid AS main_rcid, ids.foia_id AS foia_id, company_FOIA, n_us_employees, n_apps_tot, n_apps, n_success_tot, n_success, n_i129, n_uu, t, new_positions, promotions, leaves, quits, new_positions_weighted, new_positions_top3, new_positions_top10, new_positions_highn, new_hires, new_hires_weighted, new_hires_top3, new_hires_top10, new_hires_highn, n_emp, n_emp_weighted, n_emp_top3, n_emp_top10, n_emp_highn FROM (
    id_merge AS ids 
    JOIN 
    foia_for_merge AS foia
    ON ids.foia_id = foia.foia_id
    LEFT JOIN
    rev_for_merge AS rev
    ON ids.main_rcid = rev.main_rcid
)
WHERE t IS NOT NULL
""")

con.sql(f"COPY merged_for_analysis TO '{root}/data/int/merged_for_analysis_apr11.parquet'")

#####################################
# 2. BALANCED PANEL OF FIRMS (BY FOIA APP)
#       (each row is FEIN X lottery_year, sample is FEINs observed in H1B data in all 4 years)
#####################################
# step one: get balanced panel of firms (observed in all four years, drop any duplicate rcids or feins for now)
balanced_panel = con.sql("SELECT lottery_year, foia_id, FEIN FROM (SELECT FEIN, foia_id, lottery_year, COUNT(DISTINCT lottery_year) OVER(PARTITION BY FEIN) AS n_lot_years, COUNT(DISTINCT main_rcid) OVER(PARTITION BY FEIN) AS n_main_rcids FROM (SELECT *, COUNT(DISTINCT FEIN) OVER (PARTITION BY foia_id) AS n_feins FROM rmerge) WHERE n_feins = 1) WHERE n_lot_years = 4 AND n_main_rcids = 1 GROUP BY lottery_year, foia_id, FEIN")

# step two: merge with raw foia data 
balanced_samp = con.sql("SELECT *, (CASE WHEN rec_date = 'NA' THEN NULL ELSE get_fiscal_year_foia(rec_date) END) AS rec_fy, (CASE WHEN first_decision_date = 'NA' THEN NULL ELSE get_fiscal_year_foia(first_decision_date) END) AS dec_fy FROM (foia_with_ids AS a JOIN balanced_panel AS b ON a.foia_id = b.foia_id)")

# step three: group by lottery year to get n apps and successes (for treatment variable and sample restriction) at the FEIN level [note: can't use foia_for_merge because it's collapsed at the foia_id level not FEIN x lottery_year]
samp_treat = con.sql(
f"""
SELECT FEIN, lottery_year,
    COUNT(*) AS n_apps_tot,
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_apps, 
    COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_success_tot,
    COUNT(CASE WHEN status_type = 'SELECTED' AND ben_multi_reg_ind = 0 THEN 1 END) AS n_success
FROM balanced_samp 
GROUP BY FEIN, lottery_year
""")

# step four point one: group raw foia data by fiscal year received to get number of i129s received by fiscal year
foia_i129_rec = con.sql(
"""
SELECT FEIN, rec_fy, 
    COUNT(CASE WHEN ben_multi_reg_ind = 0 THEN 1 END) AS n_i129
FROM balanced_samp
WHERE rec_date != 'NA'
GROUP BY FEIN, rec_fy
""")

# step four point two: group raw foia data by dec_fy to get number of approvals by fiscal year
foia_approvals = con.sql(
"""
SELECT FEIN, dec_fy, 
    COUNT(CASE WHEN ben_multi_reg_ind = 0 AND FIRST_DECISION = 'Approved' THEN 1 END) AS n_appr
FROM balanced_samp
WHERE first_decision_date != 'NA'
GROUP BY FEIN, dec_fy
""")

# step four: merge foia data (left join onto cross join of foia_i129_rec rec_fys and FEINs to code missings as 0s)
foia_by_fy = con.sql(
"""SELECT a.y AS t, a.FEIN AS FEIN, n_i129, n_appr FROM (
    (SELECT * FROM (generate_series(2020,2024) AS x(y) CROSS JOIN (SELECT FEIN FROM balanced_panel GROUP BY FEIN))) AS a 
    LEFT JOIN 
    foia_i129_rec AS b
    ON a.y = b.rec_fy AND a.FEIN = b.FEIN
    LEFT JOIN
    foia_approvals AS c
    ON a.y = c.dec_fy AND a.FEIN = c.FEIN
    )
""")

# step five: merge together (left join onto cross join of foia_i129_rec rec_fys and FEINs to code missing n_i129s as 0s)
balanced_full = con.sql(
f"""
SELECT * FROM (
    samp_treat AS a
    LEFT JOIN 
    foia_by_fy AS b
    ON a.FEIN = b.FEIN
)
"""
)

con.sql(f"COPY balanced_full TO '{root}/data/int/balanced_full_mar25.parquet'")

