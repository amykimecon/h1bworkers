# File Description: Merging Chunked Data
# Author: Amy Kim
# Date Created: Jul 23 2025

# Imports and Paths
import pandas as pd
import os 
import time
import sys
# Ensure progress logs flush immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


# If on WRDS cloud:
if os.environ.get('USER') == 'amykimecon':
    in_path = "/home/princeton/amykimecon/data"
    out_path = "/scratch/princeton/amykimecon"

else:
    sys.path.append('../')
    from config import * 
    in_path = f'{root}/data/wrds/wrds_in'
    out_path = f'{root}/data/int'

# Iterating through Chunks
jtot = 10
chunks = []
t0 = time.time()
print(f"Merging {jtot} Chunks:")

for j in range(jtot):
    chunks = chunks + [pd.read_parquet(f"{out_path}/rev_user_merge{j}.parquet")]

chunks_merged = pd.concat(chunks)
chunks_merged.to_parquet(f"{out_path}/rev_user_merge.parquet")
t1 = time.time()

print(f"Merging Chunks Completed! Total Time Elapsed: {round((t1-t0)/60,2)} minutes")