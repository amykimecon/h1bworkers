# f1_indiv_merge

This folder is now the self-contained `f1_indiv_merge` pipeline. It owns the stage layout, root YAML config, root runner, and stage-local entrypoints.

## Stage Order

1. `01_f1_foia_clean`
2. `02_rev_import`
3. `03_rev_crosswalks`
4. `04_rev_user_clean`
5. `05_indiv_merge`

## Current State

- `01_f1_foia_clean` is now implemented locally and reads only from the root pipeline config.
- `02_rev_import` is now implemented locally: FOIA-school preprocessing, upfront WRDS `user_id` bounds scan, range-based shard planning, shard-level WRDS education scans, matched-user consolidation, and final WRDS user/position imports.
- `03_rev_crosswalks` is now implemented locally: it guards on the required cleaned education artifact from `revelio-cleaning`, stages school artifacts through `external_us_school_matching` when configured, falls back to local/legacy school and employer builders, and writes standardized stage-local crosswalk outputs including the school-family and row-level school-resolution artifacts used downstream by stage 05.
- `04_rev_user_clean` is now implemented locally: it runs stage-local `name2nat` and `nametrace` steps, blends nationality evidence with institution-country signals, applies stage-03 school/field/employer mappings, and writes cleaned user/education/position artifacts plus a match-ready exploded parquet.
- `05_indiv_merge` is now implemented locally inside `05_indiv_merge/`: it reads the root pipeline YAML, consumes stage-03 school/employer artifacts plus stage-04 cleaned Revelio artifacts, preserves the copied merge/scoring/testing logic from the current `f1_indiv_merge.py`, and runs stage-local acceptance checks after writing outputs.
- The top-level pipeline root now lives at `f1_indiv_merge/`.

## Root Config

The scaffolded root config lives at `f1_indiv_merge/pipeline.yaml`.

- It owns stage order, stage module paths, root-level build toggles, testing toggles, and the first pass of output-path contracts.
- `05_indiv_merge` now reads only from the root pipeline config; the legacy `configs/f1_indiv_merge.yaml` file is no longer needed by the stage runner.

## Inputs and Outputs

Planned stage artifacts:

- `01_f1_foia_clean`: combined raw FOIA parquet, person-id crosswalk parquet, cleaned FOIA person-panel parquet.
- `02_rev_import`: token/regex preprocessing artifacts, shard manifest, matched-user list, WRDS users parquet, WRDS positions parquet.
- `03_rev_crosswalks`: school-family crosswalk, row-level school-resolution artifact, F1 row-to-UNITID, Revelio school-to-UNITID, normalized field-to-CIP-family, row-level employer lookup, and `rcid -> normalized employer` crosswalks.
- `04_rev_user_clean`: cleaned user core, cleaned education long, cleaned position long, match-ready exploded artifact.
- `05_indiv_merge`: spell-level baseline/mult/strict outputs plus person-level baseline/strict outputs written from the local stage implementation.

## Running the Pipeline

Run the default configured stages:

```bash
python f1_indiv_merge/run_all.py
```

Run a specific stage:

```bash
python f1_indiv_merge/run_all.py --stages 05_indiv_merge
```

List the configured stages:

```bash
python f1_indiv_merge/run_all.py --list-stages
```

## Interactive Usage

Each stage has a stage-local `stage_main.py` module with a top-level `run()` entrypoint.

Example for the current working stage:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/05_indiv_merge")
import stage_main as stage
importlib.reload(stage)
stage.run()
```

For notebook-style inspection of stage 05 without writing outputs:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/05_indiv_merge")
import stage_main as stage
importlib.reload(stage)
stage.launch_ipython_session()
```

Inside `ipykernel`, that publishes `merge`, `con`, `tables`, `audit_person`, and `check_person` into the notebook namespace. `audit_person(person_id)` prints all raw join candidates before downstream filtering.

For direct interactive inspection of the copied merge logic itself:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/05_indiv_merge")
import merge_logic as merge
importlib.reload(merge)
merge.build_f1_merge_inputs(testing=True)
```

Example for the implemented `04_rev_user_clean` stage:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/04_rev_user_clean")
import stage_main as stage
importlib.reload(stage)
stage.run()
```

That stage prefers the new stage-02 WRDS user/position pulls plus the stage-03 crosswalk artifacts, but it can fall back to the legacy/current cleaned Revelio artifacts if those raw inputs are unavailable. It writes local `name2nat` and `nametrace` parquets before assembling `rev_users_core`, `rev_educ_clean_long`, `rev_pos_clean_long`, and `rev_match_ready`.

Example for the implemented crosswalk stage:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/03_rev_crosswalks")
import stage_main as stage
importlib.reload(stage)
stage.run()
```

That stage requires an external cleaned education parquet from `revelio-cleaning`. By default it falls back to the existing cleaned `rev_educ_long` artifact if present. Real-data field crosswalk builds can be expensive; raw WRDS-field enrichment is therefore config-gated via `use_raw_wrds_users_for_field_crosswalk`.

Example for the implemented FOIA stage:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/01_f1_foia_clean")
import stage_main as stage
importlib.reload(stage)
stage.run()
```

Example for the implemented `02_rev_import` stage:

```python
import importlib
import sys

sys.path.insert(0, "/home/yk0581/h1bworkers/code/f1_indiv_merge/02_rev_import")
import stage_main as stage
importlib.reload(stage)
stage.run()
```

That now completes preprocessing, a WRDS bounds scan, range shard planning, shard-level WRDS education scans, direct shard-level WRDS user/position pulls for matched users, and a final consolidation step.

Example for parallel shard scanning from separate terminals:

```bash
python f1_indiv_merge/02_rev_import/stage_main.py --scan-only --shard-id-start 0 --shard-id-end 249
python f1_indiv_merge/02_rev_import/stage_main.py --scan-only --shard-id-start 250 --shard-id-end 499
```

Each `--scan-only` worker now scans its shard range, matches users, and writes shard-local WRDS user/position chunk outputs. Then, after all shard workers finish, consolidate the existing shard outputs:

```bash
python f1_indiv_merge/02_rev_import/stage_main.py --skip-scan
```

## Current Downstream Compatibility

- `02_rev_import` now writes new pipeline-local WRDS user and position artifacts.
- `03_rev_crosswalks` now writes standardized stage-local school-family, school-resolution, school/unitid, and employer lookup artifacts. The field crosswalk is implemented and fixture-tested; the full real-data field build still needs a dedicated rerun.
- `04_rev_user_clean` now consumes the new stage-02 raw pulls when available, but it still supports legacy cleaned Revelio fallbacks so downstream work does not block on every upstream migration step.
- `05_indiv_merge` now consumes the new stage-03 and stage-04 artifacts directly. It still supports legacy Revelio fallbacks when those newer cleaned artifacts are absent, and it can optionally compare its outputs to reference/legacy parquets via config.
