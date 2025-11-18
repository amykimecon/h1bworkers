
# Imports and Paths
import wrds
import duckdb as ddb
import pandas as pd
import sys 
import os 
import time
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import re 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 
from adaptive_fuzzy.cli import run_adaptive_fuzzy_pipeline

con = ddb.connect()

# toggles
test = False
do_univ_clusters = False 
DEFAULT_RAW_EDUC_PATH = f"{root}/data/int/ihma_educ_full_raw_nov2025.parquet"

if test:
    suffix = "_test"
else:
    suffix = ""

# TEMPORARY: reading in crosswalk from wrds rsids to country/location
# db = wrds.Connection(wrds_username='amykimecon') 
# wrds_rsid_country_cw = db.raw_sql("SELECT rsid, university_country FROM revelio.individual_user_education GROUP BY rsid, university_country")
# wrds_rsid_country_cw.to_parquet(f"{root}/data/int/wrds_rsid_country_cw.parquet")

# university clustering
if do_univ_clusters:
    clusters = run_adaptive_fuzzy_pipeline(
        input = DEFAULT_RAW_EDUC_PATH,
        column = "university_raw",
        initial_labels = None,
        max_iterations = 2, max_chunk_iterations = 5,
        summarize_model = True,
        load_model = f"{root}/h1bworkers/code/adaptive_fuzzy/.adaptive_fuzzy/wrds_institutions_model.joblib",
        candidate_cache = f"{root}/data/tmp/full_candidates{suffix}.parquet",
        names_cache = f"{root}/data/tmp/full_names_cache{suffix}.parquet",
        ann_embeddings = f"{root}/data/tmp/full_embeddings{suffix}.f32",
        ann_index = f"{root}/data/tmp/full_ann{suffix}.index",
        use_ann_retrieval = True,
        build_ann_index = True,
        build_ann_embeddings = True,
        output = f"{root}/data/clean/university_clusters{suffix}.parquet",
        limit = 10000 if test else None,
        ann_top_k = 20,
        ann_nprobe = 8
    )

else:
    # import
    raw_educ_all = con.read_parquet(DEFAULT_RAW_EDUC_PATH)
    country_cw = con.read_parquet(f"{root}/data/int/wrds_rsid_country_cw.parquet")
    # clusters = con.read_parquet(f"{root}/data/clean/university_clusters{suffix}.parquet")

    # cleaning
    raw_educ_clean = con.sql(
    f"""
        WITH cleaned AS (
            SELECT
                user_id, university_name, university_raw, a.rsid, university_country, education_number, degree_raw, field_raw,
                {help.degree_clean_degree_missing_regex_sql()} AS degree_clean,
                {help.field_clean_regex_sql("field_raw")} AS field_clean,
                SUBSTRING(startdate::VARCHAR,1,4)::INT AS startyr,
                SUBSTRING(enddate::VARCHAR,1,4)::INT AS endyr,
                CASE WHEN enddate IS NOT NULL THEN SUBSTRING(enddate::VARCHAR,1,4)::INT 
                    WHEN startdate IS NOT NULL THEN SUBSTRING(startdate::VARCHAR, 1, 4)::INT + 4 
                    ELSE NULL END AS bach_gradyr
            FROM raw_educ_all AS a LEFT JOIN country_cw AS b
            ON a.rsid = b.rsid {"LIMIT 10000" if test else ""}
        )
        SELECT *, MAX(CASE WHEN degree_clean = 'Master' OR degree_clean = 'Doctor' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS has_postgrad, MAX(CASE WHEN degree_clean = 'Doctor' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS has_doctor FROM cleaned 
    """)

    # filtering user sample
    main_user_samp = con.sql(
        f"""
        SELECT * FROM (
            SELECT *,
            ROW_NUMBER() OVER(PARTITION BY user_id, degree_clean ORDER BY bach_gradyr ASC) AS edu_rank,
            MAX(CASE WHEN degree_clean = 'Bachelor' THEN 1 ELSE 0 END) OVER(PARTITION BY user_id) AS any_bach
            FROM raw_educ_clean
            WHERE (degree_clean = 'Bachelor' OR degree_clean = 'Missing') AND (NOT university_country = 'United States') AND bach_gradyr IS NOT NULL
        ) WHERE (any_bach = 1 AND degree_clean = 'Bachelor' AND edu_rank = 1) OR (any_bach = 0 AND edu_rank = 1)
        """)

    con.sql(f"COPY main_user_samp TO '{root}/data/clean/ihma_main_user_samp_nov2025{suffix}.parquet' (FORMAT PARQUET);")

    main_user_samp = con.read_parquet(f"{root}/data/clean/ihma_main_user_samp_nov2025{suffix}.parquet")

    # filtering us educ sample 
    # v1: all higher degrees
    us_user_samp = con.sql(
        f"""
        SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id, degree_clean ORDER BY startyr ASC) AS edu_rank FROM (
            SELECT * EXCLUDE(b.user_id)
            FROM raw_educ_clean AS a RIGHT JOIN (SELECT user_id, bach_gradyr AS origin_gradyr, education_number AS origin_educ_number FROM main_user_samp) AS b
            ON a.user_id = b.user_id AND a.education_number != b.origin_educ_number
        )
            WHERE (degree_clean = 'Master' OR degree_clean = 'Doctor') AND (university_country = 'United States') AND startyr IS NOT NULL AND startyr >= origin_gradyr - 1 AND startyr <= origin_gradyr + 2
        ) WHERE edu_rank = 1
        """
    )
    con.sql(f"COPY us_user_samp TO '{root}/data/clean/us_educ_samp_nov2025{suffix}.parquet' (FORMAT PARQUET);")

    us_user_samp_ma_only = con.sql(
        f"""
        SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id, degree_clean ORDER BY startyr ASC) AS edu_rank FROM (
            SELECT * EXCLUDE(b.user_id)
            FROM raw_educ_clean AS a RIGHT JOIN (SELECT user_id, bach_gradyr AS origin_gradyr, education_number AS origin_educ_number FROM main_user_samp) AS b
            ON a.user_id = b.user_id AND a.education_number != b.origin_educ_number
        )
            WHERE (degree_clean = 'Master') AND (university_country = 'United States') AND startyr IS NOT NULL AND startyr >= origin_gradyr - 1 AND startyr <= origin_gradyr + 2
        ) WHERE edu_rank = 1
        """
    )

    con.sql(f"COPY us_user_samp_ma_only TO '{root}/data/clean/us_educ_samp_ma_only_nov2025{suffix}.parquet' (FORMAT PARQUET);")

