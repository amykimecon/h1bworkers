# File Description: Merging Employer Data from Revelio and H1B Data
# Author: Amy Kim
# Date Created: Thu Feb 27

# Imports and Paths
import duckdb as ddb
import time
import employer_merge_helpers as emh
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
from splink.blocking_analysis import count_comparisons_from_blocking_rule
from splink.blocking_rule_library import CustomRule
import splink.comparison_library as cl

root = "/Users/amykim/Princeton Dropbox/Amy Kim/h1bworkers"
code = "/Users/amykim/Documents/GitHub/h1bworkers/code"

con = ddb.connect()
db_api = DuckDBAPI(con)

# # Helper Functions
# def create_replace_table(query, table_out, show = True):
#     """
#     Creates a table with a query
#     """
#     query_out = f"""
#     CREATE OR REPLACE TABLE {table_out} AS
#     {query}
#     """
#     con.execute(query_out)
#     if show:
#         con.table(table_out).show(max_rows = 10)

# Importing Data
## Importing Revelio Data
rev_raw_file = con.read_parquet(f"{root}/data/int/revelio/revelio_agg.parquet")
con.sql("CREATE TABLE rev_raw AS SELECT * FROM rev_raw_file")

## Importing FOIA Data
foia_raw_file = con.read_csv(f"{root}/data/raw/foia_bloomberg/foia_bloomberg_all.csv")
con.sql("CREATE TABLE foia_raw AS SELECT * FROM foia_raw_file")

# Cleaning Data for Merge
## filtering revelio data
emh.create_replace_table(con=con, query="SELECT * FROM rev_raw WHERE n_positions_us > 0 AND (recent_startdate_global > '2018-01-01' OR recent_enddate_global > '2018-01-01')", table_out="rev_filt", show = False)

## filtering foia data
emh.create_replace_table(con=con, query="SELECT * FROM foia_raw WHERE NOT employer_name = '(b)(3) (b)(6) (b)(7)(c)'", table_out="foia_filt", show = False)

## collapsing revelio data
### Collapsing Revelio data to company_name (clean) and rcid level, joining with rcid-level list of states and counts
rev_collapse_query = f"""
SELECT ROW_NUMBER() OVER() AS unique_id, locs.rcid as rawid, company, {emh.clean_company_string("company")} AS company_clean, n_users_us as n, pos_locations as emp_locations, pos_locations as work_locations,  naics_codes, naics2_codes, naics3_codes, naics4_codes, naics5_codes, top_naics_code, top_naics2_code, top_naics3_code, top_naics4_code, top_naics5_code
FROM (
    -- collapse all locations by rcid into list with counts
    (SELECT rcid,
        list_transform(
            list_zip(
                array_agg(state ORDER BY n_positions DESC),
                array_agg(n_positions/n_positions_us ORDER BY n_positions DESC)
            ),
            x -> struct_pack(state := x[1], share := x[2])
        ) as pos_locations
    FROM (SELECT state, rcid, sum(n_positions) AS n_positions, mean(n_positions_us) AS n_positions_us FROM rev_filt WHERE state IS NOT NULL GROUP BY state, rcid) 
    GROUP BY rcid
    ) AS locs
    FULL JOIN (
    -- collapse all NAICS codes by rcid into list with counts
    {emh.group_by_naics_code_rev()}) AS naics
    ON locs.rcid = naics.rcid
    FULL JOIN (
    -- collapse all NAICS 2-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("2")}) AS naics2
    ON locs.rcid = naics2.rcid
    FULL JOIN (
    -- collapse all NAICS 3-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("3")}) AS naics3
    ON locs.rcid = naics3.rcid
    FULL JOIN (
    -- collapse all NAICS 4-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("4")}) AS naics4
    ON locs.rcid = naics4.rcid
    FULL JOIN (
    -- collapse all NAICS 5-digit codes by rcid into list with counts
    {emh.group_by_naics_code_rev("5")}) AS naics5
    ON locs.rcid = naics5.rcid
    RIGHT JOIN (
    -- main list of companies: by company name (cleaned) and rcid (n_positions_us and n_users_us should be constant across rcids)
    SELECT rcid, 
        company, MEAN(n_positions_us) AS n_positions_us , 
        MEAN(n_users_us) AS n_users_us
    FROM rev_filt
    GROUP BY rcid, company
    ) AS raw
    ON locs.rcid = raw.rcid
)
"""
emh.create_replace_table(con, rev_collapse_query, "rev_collapsed", show = False)

## collapsing foia data
### Collapsing FOIA data to company_name (clean) and FEIN level,
###     joining with FEIN-level list of states and counts, FEIN-level list of worksite states and counts, FEIN-level list of NAICS codes and counts
foia_collapse_query = f"""
WITH employer_counts AS (
    SELECT FEIN, state_name, COUNT(CASE WHEN NOT state_name = 'NA' THEN 1 END) OVER (PARTITION BY FEIN) AS n_apps_employer_tot, 
    worksite_state_name, COUNT(CASE WHEN NOT worksite_state_name = 'NA' THEN 1 END) OVER (PARTITION BY FEIN) AS n_wins_worksite_tot, 
    NAICS_CODE, SUBSTR(NAICS_CODE, 1, 2) AS NAICS2_CODE, SUBSTR(NAICS_CODE, 1, 3) AS NAICS3_CODE, SUBSTR(NAICS_CODE, 1, 4) AS NAICS4_CODE, SUBSTR(NAICS_CODE, 1, 5) AS NAICS5_CODE, COUNT(CASE WHEN NOT NAICS_CODE = 'NA' AND NOT NAICS_CODE = '999999' THEN 1 END) OVER (PARTITION BY FEIN) AS n_wins_naics_tot FROM foia_filt) 
SELECT ROW_NUMBER() OVER() AS unique_id, apps.FEIN as rawid, company, {emh.clean_company_string("company")} AS company_clean, n_us_employees AS n, emp_locations, work_locations, naics_codes, naics2_codes, naics3_codes, naics4_codes, naics5_codes, top_naics_code, top_naics2_code, top_naics3_code, top_naics4_code, top_naics5_code
FROM (
    -- collapse all locations by FEIN into list with counts
    (SELECT FEIN, 
        list_transform(
            list_zip(
                array_agg(state_name ORDER BY n_apps_employer_state DESC), 
                array_agg(n_apps_employer_state/n_apps_employer_tot ORDER BY n_apps_employer_state DESC)
            ), 
            x -> struct_pack(state := x[1], share := x[2])
        ) as emp_locations
    FROM (
        -- grouping by FEIN and state
        SELECT FEIN, state_name, MEAN(n_apps_employer_tot) AS n_apps_employer_tot, COUNT(*) AS n_apps_employer_state FROM employer_counts WHERE NOT state_name = 'NA' GROUP BY FEIN, state_name)
    GROUP BY FEIN
    ) AS apps
    FULL JOIN (
    -- collapse all worksites by FEIN into list with counts
    SELECT FEIN, 
        list_transform(
            list_zip(
                array_agg(worksite_state_name ORDER BY n_wins_worksite_state DESC), 
                array_agg(n_wins_worksite_state/n_wins_worksite_tot ORDER BY n_wins_worksite_state DESC)
            ), 
            x -> struct_pack(state := x[1], share := x[2])
        ) as work_locations
    FROM (
        -- grouping by FEIN and state
        SELECT FEIN, worksite_state_name, MEAN(n_wins_worksite_tot) AS n_wins_worksite_tot, COUNT(*) AS n_wins_worksite_state FROM employer_counts WHERE NOT worksite_state_name = 'NA' GROUP BY FEIN, worksite_state_name)
    GROUP BY FEIN
    ) AS worksites
    ON apps.FEIN = worksites.FEIN
    FULL JOIN (
    -- collapse all NAICS codes by FEIN into list with counts
    {emh.group_by_naics_code_foia()}) AS naics
    ON apps.FEIN = naics.FEIN
    FULL JOIN (
    -- collapse all NAICS 2-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("2")}) AS naics2
    ON apps.FEIN = naics2.FEIN
    FULL JOIN (
    -- collapse all NAICS 3-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("3")}) AS naics3
    ON apps.FEIN = naics3.FEIN
    FULL JOIN (
    -- collapse all NAICS 4-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("4")}) AS naics4
    ON apps.FEIN = naics4.FEIN
    FULL JOIN (
    -- collapse all NAICS 5-digit codes by FEIN into list with counts
    {emh.group_by_naics_code_foia("5")}) AS naics5
    ON apps.FEIN = naics5.FEIN
    RIGHT JOIN (
    -- main list of employers: by company name (cleaned) and FEIN
    SELECT FEIN, 
        employer_name AS company, MAX(CASE WHEN NOT NUM_OF_EMP_IN_US = 'NA' THEN NUM_OF_EMP_IN_US::INTEGER END) AS n_us_employees,
        COUNT(*) AS n_apps, COUNT(CASE WHEN status_type = 'SELECTED' THEN 1 END) AS n_wins,
    FROM foia_filt
    GROUP BY FEIN, company
    ) AS raw
    ON apps.FEIN = raw.FEIN
)
"""
emh.create_replace_table(con,foia_collapse_query, "foia_collapsed", show = False)

## tokenizing both datasets
emh.tokenize(con, "rev_collapsed", "rev_tokenized", "company", "\\s+", show = True)
emh.tokenize(con, "foia_collapsed", "foia_tokenized", "company", "\\s+", show = True)
# emh.create_replace_table(con, "SELECT *, 'rev' AS dataset FROM rev_collapsed UNION ALL SELECT *, 'foia' AS dataset FROM foia_collapsed", "all_collapsed")
# emh.tokenize(con, "all_collapsed", "all_tokenized", "company", "\\s+", show = True)

# TESTING BLOCKING RULES
# ## intersecting rare tokens
# counts = count_comparisons_from_blocking_rule(
#     table_or_tables=[con.sql("SELECT * FROM rev_tokenized LIMIT 10000"), con.sql("SELECT * FROM foia_tokenized")],
#     blocking_rule = CustomRule("LENGTH(LIST_INTERSECT(l.rare_name_tokens, r.rare_name_tokens)) > 0"),                 
#     db_api=db_api, link_type = "link_only",
#     max_rows_limit=1e20)
# print(counts)

# counts = count_comparisons_from_blocking_rule(
#     table_or_tables=[con.sql("SELECT * FROM rev_tokenized"), con.sql("SELECT * FROM foia_tokenized")],
#     blocking_rule = CustomRule("((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token))"),                 
#     db_api=db_api, link_type = "link_only",
#     max_rows_limit=1e20)
# print(counts)

# # ## state & name
# # counts = count_comparisons_from_blocking_rule(
# #     table_or_tables=[con.sql("SELECT * FROM rev_tokenized LIMIT 10000"), con.sql("SELECT * FROM foia_tokenized")],
# #     blocking_rule = CustomRule(""),                 
# #     db_api=db_api, link_type = "link_only",
# #     max_rows_limit=1e20)
# # print(counts)

# # making test tables
# emh.create_replace_table(con, "SELECT * FROM foia_tokenized LIMIT 10000", "foia_test")
# # emh.create_replace_table(con, "SELECT * FROM foia_tokenized WHERE dataset = 'foia' AND REGEXP_MATCHES(company_clean, 'microsoft|uber|wells|blockchain')", "foia_test")
# con.sql("SELECT company_clean FROM foia_test").df()
# # emh.create_replace_table(con, "SELECT * FROM all_tokenized WHERE dataset = 'rev' AND REGEXP_MATCHES(company_clean, 'microsoft|uber|wells|blockchain')", "rev_test")

# creating joined test table with blocking rules
# join_query = f"SELECT * FROM rev_tokenized as l JOIN foia_test as r on ((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))"

# # testing name token join
# con.sql(f"SELECT l.company_clean, r.company_clean, l.rawid, r.rawid, {emh.mult_prod_matching_freqs('name_tokens_with_freq','token','freq')} AS prod FROM rev_tokenized as l JOIN foia_test as r on ((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))").df()

# comparison of naics codes
name_comp = {
"output_column_name": "name_tokens_with_freq",
"comparison_levels": [
    {
        "sql_condition": '"name_tokens_with_freq_l" IS NULL OR "name_tokens_with_freq_r" IS NULL',
        "label_for_charts": "name_tokens_with_freq is NULL",
        "is_null_level": True,
    },
    {
        "sql_condition": f"""
        {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-18
        """,
        "label_for_charts": "Array product is less than 1e-18",
    },
    {
        "sql_condition": f"""
        {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-15
        """,
        "label_for_charts": "Array product is less than 1e-15",
    },
    {
        "sql_condition": f"""
        {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-12
        """,
        "label_for_charts": "Array product is less than 1e-12",
    },
    {
        "sql_condition": f"""
        {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-9
        """,
        "label_for_charts": "Array product is less than 1e-9",
    },
    {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
    ],
"comparison_description": "Comparison of levels of product of frequencies of exactly matched name tokens",
}

# TODO: add column for tf adjustment of locations
locs_comp = {
"output_column_name": "all_locations",
"comparison_levels": [
    {
        "sql_condition": '("emp_locations_l" IS NULL OR "emp_locations_r" IS NULL) AND ("work_locations_l" IS NULL OR "work_locations_r" IS NULL)',
        "label_for_charts": "Null",
        "is_null_level": True
    },
    {
        "sql_condition": f"""
        {emh.dot_prod_matching_freqs("work_locations", "state","share")} > 0.8
        """,
        "label_for_charts": "Match on Work Locations Over 0.8"
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0.8""",
        "label_for_charts": "Match on Employer Locations Over 0.8"
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('work_locations', 'state','share')} > 0.2""",
        "label_for_charts": "Match on Work Locations Over 0.2"
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0.2""",
        "label_for_charts": "Match on Employer Locations Over 0.2"
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('work_locations', 'state','share')} > 0""",
        "label_for_charts": "Match on Work Locations Over 0"
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0""",
        "label_for_charts": "Match on Employer Locations Over 0"
    },
    {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"}
    ],
    "comparison_description": "Degree of match on work or employer states"
}

naics_comp = {
"output_column_name": "naics_codes",
"comparison_levels": [
    {
        "sql_condition": '("naics_codes_l" IS NULL OR "naics_codes_r" IS NULL) AND ("naics5_codes_l" IS NULL OR "naics5_codes_r" IS NULL) AND ("naics4_codes_l" IS NULL OR "naics4_codes_r" IS NULL) AND ("naics3_codes_l" IS NULL OR "naics3_codes_r" IS NULL) AND ("naics2_codes_l" IS NULL OR "naics2_codes_r" IS NULL)',
        "label_for_charts": "Null",
        "is_null_level": True
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('naics_codes', 'naics', 'share')} > 0.9""",
        "label_for_charts": "Match on Full NAICS Code",
        "tf_adjustment_column": "top_naics_code"
    },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('naics5_codes', 'naics', 'share')} > 0.9""",
    #     "label_for_charts": "Match on 5-digit NAICS Code",
    #     "tf_adjustment_column": "top_naics5_code"
    # },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('naics4_codes', 'naics', 'share')} > 0.9""",
        "label_for_charts": "Match on 4-digit NAICS Code",
        "tf_adjustment_column": "top_naics4_code"
    },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('naics3_codes', 'naics', 'share')} > 0.9""",
    #     "label_for_charts": "Match on 3-digit NAICS Code",
    #     "tf_adjustment_column": "top_naics3_code"
    # },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('naics2_codes', 'naics', 'share')} > 0.9""",
        "label_for_charts": "Match on 2-digit NAICS Code",
        "tf_adjustment_column": "top_naics2_code"
    },
    {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
    ],
    "comparison_description": "Degree of match on different digits of NAICS codes"
}



settings = SettingsCreator(
    link_type="link_only",
    blocking_rules_to_generate_predictions= [CustomRule("((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))")],
    comparisons = [
        name_comp, 
        locs_comp,
        naics_comp],
    retain_intermediate_calculation_columns=True,
    retain_matching_columns=True
)

# TEN ROWS, DEFAULT MAX PAIRS
start = time.time()
linker = Linker([con.sql("SELECT * FROM rev_tokenized"), 
                 con.sql("SELECT * FROM foia_tokenized LIMIT 10")], settings, db_api)

linker.training.estimate_u_using_random_sampling()

end = time.time()
print(f"Time to train model (ten rows, default max pairs): {end-start}")

# TRAINING USING EM
start = time.time()
blocking_for_training = block_on()
linker.training.estimate_u_using_random_sampling()

end = time.time()
print(f"Time to train model (ten rows, default max pairs): {end-start}")

#linker.visualisations.match_weights_chart()

# ONE THOUSAND ROWS, DEFAULT MAX PAIRS
start = time.time()
linker2 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
                 con.sql("SELECT * FROM foia_tokenized LIMIT 1000")], settings, db_api)

linker2.training.estimate_u_using_random_sampling()

end = time.time()
print(f"Time to train model (1000 rows, default max pairs): {end-start}")

linker2.visualisations.match_weights_chart()

# TEN ROWS, 1e9 MAX PAIRS
start = time.time()
linker3 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
                 con.sql("SELECT * FROM foia_tokenized LIMIT 10")], settings, db_api)

linker3.training.estimate_u_using_random_sampling(max_pairs=1e9)

end = time.time()
print(f"Time to query and save: {end-start}")

linker3.visualisations.match_weights_chart()

# ONE THOUSAND ROWS, 1e9 MAX PAIRS
start = time.time()
linker4 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
                 con.sql("SELECT * FROM foia_tokenized LIMIT 1000")], settings, db_api)

linker4.training.estimate_u_using_random_sampling(max_pairs=1e9)

end = time.time()
print(f"Time to query and save: {end-start}")

linker4.visualisations.match_weights_chart()


#linker.visualisations.m_u_parameters_chart()

# linker2 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
#                  con.sql("SELECT * FROM foia_tokenized LIMIT 10000")], settings, db_api)

# linker2.training.estimate_u_using_random_sampling(max_pairs = 1e9)

# linker2.visualisations.match_weights_chart()

# linker2.visualisations.m_u_parameters_chart()

# settings = SettingsCreator(
#     link_type="link_only",
#     blocking_rules_to_generate_predictions= [CustomRule("((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))")],
#     comparisons = [
#         name_comp,
#         cl.JaroAtThresholds("company_clean")],
#     retain_intermediate_calculation_columns=True,
#     retain_matching_columns=True
# )

# linker = Linker([con.sql("SELECT * FROM rev_tokenized"), 
#                  con.sql("SELECT * FROM foia_test")], settings, db_api)

# linker.training.estimate_u_using_random_sampling()

# linker.visualisations.match_weights_chart()

# linker.visualisations.m_u_parameters_chart()

# con.sql("SELECT l.company_clean, r.company_clean, list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq), list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq).list_transform(x -> x.freq::float), list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq).list_transform(x -> x.freq::float).list_concat([1.0::FLOAT]), list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq).list_transform(x -> x.freq::float).list_concat([1.0::FLOAT]).list_reduce((p,q) -> p*q) AS product, l.name_tokens_with_freq, r.name_tokens_with_freq FROM rev_test as l JOIN foia_test as r on ((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token))").df()
