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

#####################
# IMPORTING DATA
#####################
## Importing Revelio Data
rev_tok_file = con.read_parquet(f"{root}/data/int/splink/rev_tokenized.parquet")
con.sql("CREATE OR REPLACE TABLE rev_tokenized AS SELECT * FROM rev_tok_file")

## Importing FOIA Data
foia_tok_file = con.read_parquet(f"{root}/data/int/splink/foia_tokenized.parquet")
con.sql("CREATE OR REPLACE TABLE foia_tokenized AS SELECT * FROM foia_tok_file")

## Importing All Data (tokenized jointly)
all_tok_file = con.read_parquet(f"{root}/data/int/splink/all_tokenized.parquet")
con.sql("CREATE OR REPLACE TABLE all_tokenized AS SELECT * FROM all_tok_file")

# foiadf = con.sql("SELECT * FROM foia_tokenized")
# revdf = con.sql("SELECT * FROM rev_tokenized")
foiadf = con.sql("SELECT * FROM all_tokenized WHERE dataset = 'foia'")# LIMIT 10000")
revdf = con.sql("SELECT * FROM all_tokenized WHERE dataset = 'rev'")# LIMIT 500000")

#################################
# DEFINING COMPARISONS FOR SPLINK
#################################
tf_adj_wt = 1
tf_min_u_val = 0.001
# getting mean/median token frequency across foia and rev
foia_name_token_avg_freq = con.sql("SELECT SUM(freqsum)/SUM(freqcount) AS freq FROM (SELECT list_transform(name_tokens_with_freq, x -> x.freq).list_sum() AS freqsum, list_transform(name_tokens_with_freq, x -> x.freq).list_count() AS freqcount FROM 'foiadf')").df()["freq"][0]
foia_name_token_med_freq = con.sql("SELECT MEDIAN(freq) AS freq FROM (SELECT list_transform(name_tokens_with_freq, x -> x.freq).list_median() AS freq FROM 'foiadf')").df()["freq"][0]

rev_name_token_avg_freq = con.sql("SELECT SUM(freqsum)/SUM(freqcount) AS freq FROM (SELECT list_transform(name_tokens_with_freq, x -> x.freq).list_sum() AS freqsum, list_transform(name_tokens_with_freq, x -> x.freq).list_count() AS freqcount FROM 'revdf')").df()["freq"][0]
rev_name_token_med_freq = con.sql("SELECT MEDIAN(freq) AS freq FROM (SELECT list_transform(name_tokens_with_freq, x -> x.freq).list_median() AS freq FROM 'revdf')").df()["freq"][0]

freq_prod = foia_name_token_avg_freq*rev_name_token_avg_freq
print(f"Product of average FOIA and rev token freqs: {freq_prod}")
med_freq_prod = foia_name_token_med_freq*rev_name_token_med_freq
print(f"Product of median FOIA and rev token freqs: {med_freq_prod}")

# comparison of tokenized names with frequencies
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
    # {
    #     "sql_condition": f"""
    #     {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-15
    #     """,
    #     "label_for_charts": "Array product is less than 1e-15",
    # },
    # {
    #     "sql_condition": f"""
    #     {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-12
    #     """,
    #     "label_for_charts": "Array product is less than 1e-12",
    # },
    {
        "sql_condition": f"""
        {emh.mult_prod_matching_freqs('name_tokens_with_freq', 'token', 'freq')} < 1e-10
        """,
        "label_for_charts": "Array product is less than 1e-10",
    },
    {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
    ],
"comparison_description": "Comparison of levels of product of frequencies of exactly matched name tokens",
}

locs_comp = {
"output_column_name": "all_locations",
"comparison_levels": [
    {
        "sql_condition": '("emp_locations_l" IS NULL OR "emp_locations_r" IS NULL) AND ("work_locations_l" IS NULL OR "work_locations_r" IS NULL)',
         # AND ("top_emp_location_l" IS NULL OR "top_emp_location_r" IS NULL)',
        "label_for_charts": "Null",
        "is_null_level": True
    },
    {
        "sql_condition": f"""
        {emh.dot_prod_matching_freqs("work_locations", "state","share")} > 0.8 OR {emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0.8 
        """,
        "label_for_charts": "Match on Locations Over 0.8",
        # "tf_adjustment_column": "top_emp_location",
        # "tf_adjustment_weight": {tf_adj_wt},
        # "tf_minimum_u_value": {tf_min_u_val}
    },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0.8""",
    #     "label_for_charts": "Match on Employer Locations Over 0.8"
    # },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('work_locations', 'state','share')} > 0.2 OR {emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0.2""",
    #     "label_for_charts": "Match on Locations Over 0.2",
    #     # "tf_adjustment_column": "top_emp_location",
    #     # "tf_adjustment_weight": {tf_adj_wt},
    #     # "tf_minimum_u_value": {tf_min_u_val}
    # },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0.2""",
    #     "label_for_charts": "Match on Employer Locations Over 0.2"
    # },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('work_locations', 'state','share')} > 0 OR {emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0""",
        "label_for_charts": "Match on Work Locations Over 0",
        # "tf_adjustment_column": "top_emp_location",
        # "tf_adjustment_weight": {tf_adj_wt},
        # "tf_minimum_u_value": {tf_min_u_val}
    },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('emp_locations', 'state','share')} > 0""",
    #     "label_for_charts": "Match on Employer Locations Over 0"
    # },
    {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"}
    ],
    "comparison_description": "Degree of match on work or employer states"
}

naics_comp = {
"output_column_name": "naics_codes",
"comparison_levels": [
    {
        "sql_condition": '("naics_codes_l" IS NULL OR "naics_codes_r" IS NULL) AND ("naics5_codes_l" IS NULL OR "naics5_codes_r" IS NULL) AND ("naics4_codes_l" IS NULL OR "naics4_codes_r" IS NULL) AND ("naics3_codes_l" IS NULL OR "naics3_codes_r" IS NULL) AND ("naics2_codes_l" IS NULL OR "naics2_codes_r" IS NULL)',
        #AND ("top_naics_code_l" IS NULL OR "top_naics_code_r" IS NULL) AND ("top_naics5_code_l" IS NULL OR "top_naics5_code_r" IS NULL) AND("top_naics4_code_l" IS NULL OR "top_naics4_code_r" IS NULL) AND("top_naics3_code_l" IS NULL OR "top_naics3_code_r" IS NULL) AND ("top_naics2_code_l" IS NULL OR "top_naics2_code_r" IS NULL)',
        "label_for_charts": "Null",
        "is_null_level": True
    },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('naics_codes', 'naics', 'share')} > 0.9""",
        "label_for_charts": "Match on Full NAICS Code",
        # "tf_adjustment_column": "top_naics_code",
        # "tf_adjustment_weight": {tf_adj_wt},
        # "tf_minimum_u_value": {tf_min_u_val}
    },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('naics5_codes', 'naics', 'share')} > 0.9""",
    #     "label_for_charts": "Match on 5-digit NAICS Code",
    #     "tf_adjustment_column": "top_naics5_code"
    # },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('naics4_codes', 'naics', 'share')} > 0.9""",
    #     "label_for_charts": "Match on 4-digit NAICS Code",
    #     # "tf_adjustment_column": "top_naics4_code",
    #     # "tf_adjustment_weight": {tf_adj_wt},
    #     # "tf_minimum_u_value": {tf_min_u_val}
    # },
    # {
    #     "sql_condition": f"""{emh.dot_prod_matching_freqs('naics3_codes', 'naics', 'share')} > 0.9""",
    #     "label_for_charts": "Match on 3-digit NAICS Code",
    #     "tf_adjustment_column": "top_naics3_code"
    # },
    {
        "sql_condition": f"""{emh.dot_prod_matching_freqs('naics2_codes', 'naics', 'share')} > 0.9""",
        "label_for_charts": "Match on 2-digit NAICS Code",
        # "tf_adjustment_column": "top_naics2_code",
        # "tf_adjustment_weight": {tf_adj_wt},
        # "tf_minimum_u_value": {tf_min_u_val}
    },
    {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
    ],
    "comparison_description": "Degree of match on different digits of NAICS codes"
}


#################################
# INITIALIZING SPLINK MODEL
#################################
settings = SettingsCreator(
    link_type="link_only",
    blocking_rules_to_generate_predictions= [
        CustomRule("l.rarest_token = r.rarest_token AND l.second_rarest_token = r.second_rarest_token"),
        CustomRule("l.rarest_token = r.second_rarest_token AND l.second_rarest_token = r.rarest_token")],
    comparisons = [
        name_comp, 
        locs_comp,
        naics_comp],
    retain_intermediate_calculation_columns=True,
    retain_matching_columns=True
)

linker = Linker([foiadf, revdf], settings, db_api)

settings_alt = SettingsCreator(
    link_type="link_only",
    blocking_rules_to_generate_predictions= [
        CustomRule("l.rarest_token = r.rarest_token AND l.second_rarest_token = r.second_rarest_token"),
        CustomRule("l.rarest_token = r.second_rarest_token AND l.second_rarest_token = r.rarest_token")],
    comparisons = [
        cl.JaroAtThresholds("company_clean"),
        cl.ExactMatch("top_emp_location"),
        cl.ExactMatch("top_naics_code")],
    retain_intermediate_calculation_columns=True,
    retain_matching_columns=True
)

linker_alt = Linker([foiadf, revdf], settings_alt, db_api)

settings2 = SettingsCreator(
    link_type="link_and_dedupe",
    blocking_rules_to_generate_predictions= [
        CustomRule("l.rarest_token = r.rarest_token AND l.second_rarest_token = r.second_rarest_token"),
        CustomRule("l.rarest_token = r.second_rarest_token AND l.second_rarest_token = r.rarest_token")],
    comparisons = [
        name_comp, 
        locs_comp,
        naics_comp],
    retain_intermediate_calculation_columns=True,
    retain_matching_columns=True
)

linker2 = Linker([foiadf, revdf], settings2, db_api)

settings2_alt = SettingsCreator(
    link_type="link_and_dedupe",
    blocking_rules_to_generate_predictions= [
        CustomRule("l.rarest_token = r.rarest_token AND l.second_rarest_token = r.second_rarest_token"),
        CustomRule("l.rarest_token = r.second_rarest_token AND l.second_rarest_token = r.rarest_token")],
    comparisons = [
        cl.JaroAtThresholds("company_clean"),
        cl.ExactMatch("top_emp_location"),
        cl.ExactMatch("top_naics_code")],
    retain_intermediate_calculation_columns=True,
    retain_matching_columns=True
)

linker2_alt = Linker([foiadf, revdf], settings2_alt, db_api)

#################################
# TRAINING MODEL
#################################
# estimating lambdas
linker.training.estimate_probability_two_random_records_match([block_on("company_clean")], recall = 0.7)

linker_alt.training.estimate_probability_two_random_records_match([block_on("company_clean")], recall = 0.7)

linker2.training.estimate_probability_two_random_records_match([block_on("company_clean")], recall = 0.7)

linker2_alt.training.estimate_probability_two_random_records_match([block_on("company_clean")], recall = 0.7)

#estimating us
start = time.time()
linker.training.estimate_u_using_random_sampling(max_pairs = 1e7)
end = time.time()
print(f"time to estimate u for model 1 (link only): {end-start}")

start = time.time()
linker_alt.training.estimate_u_using_random_sampling(max_pairs = 1e7)
end = time.time()
print(f"time to estimate u for model 2 (link only): {end-start}")

start = time.time()
linker2.training.estimate_u_using_random_sampling(max_pairs = 1e7)
end = time.time()
print(f"time to estimate u for model 1 (link + dedupe): {end-start}")

start = time.time()
linker2_alt.training.estimate_u_using_random_sampling(max_pairs = 1e7)
end = time.time()
print(f"time to estimate u for model 2 (link + dedupe): {end-start}")

# estimating m: note - need to fix so it's not estimating m for name on names etc.
## blocking on name
start = time.time()
linker_alt.training.estimate_parameters_using_expectation_maximisation(block_on("company_clean"))

# blocking on top employer location and naics code
linker_alt.training.estimate_parameters_using_expectation_maximisation(block_on("top_emp_location", "top_naics_code"))
end = time.time()
print(f"time to train model 2 (link only): {end-start}")

## blocking on name
start = time.time()
linker2_alt.training.estimate_parameters_using_expectation_maximisation(block_on("company_clean"))

# blocking on top employer location and naics code
linker2_alt.training.estimate_parameters_using_expectation_maximisation(block_on("top_emp_location", "top_naics_code"))
end = time.time()
print(f"time to train model 2 (link + dedupe): {end-start}")

# saving models
linker.misc.save_model_to_json(f"{root}/data/int/splink/linker_base.json", overwrite = True)

linker_alt.misc.save_model_to_json(f"{root}/data/int/splink/linker_simp.json", overwrite = True)

linker2.misc.save_model_to_json(f"{root}/data/int/splink/linker_dedupe_base.json", overwrite = True)

linker2_alt.misc.save_model_to_json(f"{root}/data/int/splink/linker_dedupe_simp.json", overwrite = True)

#################################
# EVALUATING MODEL
#################################
# linker.visualisations.match_weights_chart()

# linker.visualisations.m_u_parameters_chart()

df_predict = linker.inference.predict()
df_predict_alt = linker_alt.inference.predict()
df_predict2 = linker2.inference.predict()
df_predict2_alt = linker2_alt.inference.predict()


#linker.visualisations.match_weights_histogram(df_predict)

# counting 
for df in [df_predict, df_predict_alt, df_predict2, df_predict2_alt]:
    fullct = con.sql(f"SELECT COUNT(*) AS ct FROM {df.physical_name}").df()['ct'][0]
    matchct = con.sql(f"SELECT COUNT(*) AS ct FROM {df.physical_name} WHERE match_probability > 0.9").df()['ct'][0]
    print(f"{matchct} good matches out of {fullct} pairs")

#df_predict_pd = df_predict.as_pandas_dataframe()

# Linking back to original dfs
# con.sql(f"SELECT * FROM (
#         SELECT unique_id_l, unique_id_l FROM {df_predict.physical_name}) as matches LEFT JOIN (SELECT unique_id, company_clean, )")



# #################################
# # TESTING BLOCKING RULES
# #################################
# # RULE ONE: exact match of cleaned company string (use as deterministic rule for lambda)
# counts1 = count_comparisons_from_blocking_rule(
#     table_or_tables=[foiadf, revdf],
#     blocking_rule = block_on("company_clean"),
#     link_type = "link_only",
#     db_api = db_api
# )
# print(counts1)

# # RULE TWO: 
# counts2a = count_comparisons_from_blocking_rule(
#     table_or_tables=[foiadf, revdf],
#     blocking_rule = CustomRule("l.rarest_token = r.rarest_token AND l.second_rarest_token = r.second_rarest_token"),                 
#     db_api=db_api, link_type = "link_only")
# counts2b = count_comparisons_from_blocking_rule(
#     table_or_tables=[foiadf, revdf],
#     blocking_rule = CustomRule("l.rarest_token = r.second_rarest_token AND l.second_rarest_token = r.rarest_token"),                 
#     db_api=db_api, link_type = "link_only")
# print(counts2a)
# print(counts2b)

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
# con.sql(f"SELECT l.company_clean, r.company_clean, {emh.mult_prod_matching_freqs('name_tokens_with_freq','token','freq', dotnote=True)} AS prod FROM 'foiadf' as l JOIN 'revdf' as r on ((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))").df()



# settings = SettingsCreator(
#     link_type="link_only",
#     blocking_rules_to_generate_predictions= [CustomRule("((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))")],
#     comparisons = [
#         name_comp, 
#         locs_comp,
#         naics_comp],
#     retain_intermediate_calculation_columns=True,
#     retain_matching_columns=True
# )

# # TEN ROWS, DEFAULT MAX PAIRS
# start = time.time()
# linker = Linker([con.sql("SELECT * FROM rev_tokenized"), 
#                  con.sql("SELECT * FROM foia_tokenized LIMIT 10")], settings, db_api)

# linker.training.estimate_u_using_random_sampling()

# end = time.time()
# print(f"Time to train model (ten rows, default max pairs): {end-start}")

# # TRAINING USING EM
# start = time.time()
# blocking_for_training = block_on()
# linker.training.estimate_u_using_random_sampling()

# end = time.time()
# print(f"Time to train model (ten rows, default max pairs): {end-start}")

# #linker.visualisations.match_weights_chart()

# # ONE THOUSAND ROWS, DEFAULT MAX PAIRS
# start = time.time()
# linker2 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
#                  con.sql("SELECT * FROM foia_tokenized LIMIT 1000")], settings, db_api)

# linker2.training.estimate_u_using_random_sampling()

# end = time.time()
# print(f"Time to train model (1000 rows, default max pairs): {end-start}")

# linker2.visualisations.match_weights_chart()

# # TEN ROWS, 1e9 MAX PAIRS
# start = time.time()
# linker3 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
#                  con.sql("SELECT * FROM foia_tokenized LIMIT 10")], settings, db_api)

# linker3.training.estimate_u_using_random_sampling(max_pairs=1e9)

# end = time.time()
# print(f"Time to query and save: {end-start}")

# linker3.visualisations.match_weights_chart()

# # ONE THOUSAND ROWS, 1e9 MAX PAIRS
# start = time.time()
# linker4 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
#                  con.sql("SELECT * FROM foia_tokenized LIMIT 1000")], settings, db_api)

# linker4.training.estimate_u_using_random_sampling(max_pairs=1e9)

# end = time.time()
# print(f"Time to query and save: {end-start}")

# linker4.visualisations.match_weights_chart()


# #linker.visualisations.m_u_parameters_chart()

# # linker2 = Linker([con.sql("SELECT * FROM rev_tokenized"), 
# #                  con.sql("SELECT * FROM foia_tokenized LIMIT 10000")], settings, db_api)

# # linker2.training.estimate_u_using_random_sampling(max_pairs = 1e9)

# # linker2.visualisations.match_weights_chart()

# # linker2.visualisations.m_u_parameters_chart()

# # settings = SettingsCreator(
# #     link_type="link_only",
# #     blocking_rules_to_generate_predictions= [CustomRule("((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token)) OR ((l.rarest_token = r.rarest_token OR l.rarest_token = r.second_rarest_token OR r.rarest_token = l.second_rarest_token) AND (l.second_rarest_token IS NULL OR r.second_rarest_token IS NULL))")],
# #     comparisons = [
# #         name_comp,
# #         cl.JaroAtThresholds("company_clean")],
# #     retain_intermediate_calculation_columns=True,
# #     retain_matching_columns=True
# # )

# # linker = Linker([con.sql("SELECT * FROM rev_tokenized"), 
# #                  con.sql("SELECT * FROM foia_test")], settings, db_api)

# # linker.training.estimate_u_using_random_sampling()

# # linker.visualisations.match_weights_chart()

# # linker.visualisations.m_u_parameters_chart()

# # con.sql("SELECT l.company_clean, r.company_clean, list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq), list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq).list_transform(x -> x.freq::float), list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq).list_transform(x -> x.freq::float).list_concat([1.0::FLOAT]), list_intersect(l.name_tokens_with_freq, r.name_tokens_with_freq).list_transform(x -> x.freq::float).list_concat([1.0::FLOAT]).list_reduce((p,q) -> p*q) AS product, l.name_tokens_with_freq, r.name_tokens_with_freq FROM rev_test as l JOIN foia_test as r on ((l.rarest_token = r.rarest_token) AND (l.second_rarest_token = r.second_rarest_token)) OR ((l.rarest_token = r.second_rarest_token) AND (l.second_rarest_token = r.rarest_token))").df()
