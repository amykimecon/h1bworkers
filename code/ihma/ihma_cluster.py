# Pre-processing raw WRDS data to get university clusters and CIP code matches

import duckdb as ddb
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
import adaptive_fuzzy.cli as af_cli
from adaptive_fuzzy.cli import run_adaptive_fuzzy_pipeline

def _no_expand(self, subset_size, max_candidates=None):
    print(f"[FAST] skipping expand_to_size({subset_size:,})")
    return

af_cli.CandidateCacheManager.expand_to_size = _no_expand

test = False

# Running adaptive fuzzy to get university clusters
if test:
    suffix = "_test"
else:
    suffix = ""

clusters = run_adaptive_fuzzy_pipeline(
    input = f"{root}/data/int/ihma_users_all_nov2025.parquet",
    column = "university_raw",
    initial_labels = None,
    max_iterations = 0, max_chunk_iterations = 0,
    summarize_model = True,
    load_model = f"{root}/h1bworkers/code/adaptive_fuzzy/.adaptive_fuzzy/wrds_institutions_model.joblib",
    candidate_cache = f"{root}/data/tmp/ihma_candidates{suffix}.parquet",
    names_cache = f"{root}/data/tmp/ihma_names_cache{suffix}.parquet",
    ann_embeddings = f"{root}/data/tmp/ihma_embeddings{suffix}.f32",
    ann_index = f"{root}/data/tmp/ihma_ann{suffix}.index",
    use_ann_retrieval = True,
    output = f"{root}/data/int/ihma_clusters_nov2025{suffix}.parquet",
    limit = 10000 if test else None,
    ann_top_k = 20,
    ann_nprobe = 8
)

# 