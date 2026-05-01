# Source-Only OPT Exposure / Event-Study Pipeline

This README documents the current OPT exposure workflow in `company_shift_share`.

This is the pipeline that:

- builds firm-level OPT outcomes directly from source FOIA data,
- builds firm-level Revelio covariates directly from WRDS,
- samples outside-negative firms directly from Revelio company mapping,
- fits an `opt_probability_index` exposure model,
- runs the final event study on an in-house analysis panel,
- and optionally runs a testing-only diagnostic mode that skips event-study estimation.

The important design choice is that this workflow does **not** depend on outputs from `build_company_shift_share.py`. The older company shift-share instrument code still exists in this folder, but it is not upstream of this pipeline.

## Files That Matter

- `company_shift_share/source_exposure_data.py`
  Builds the shared source-data objects and WRDS caches.
- `company_shift_share/revelio_company_features.py`
  Builds the firm-level pre-period feature matrix used for the exposure model.
- `company_shift_share/exposure_event_study.py`
  Fits the exposure model, saves predictions, and runs the event study.
- `configs/company_shift_share_exposure_event_study.yaml`
  Minimal config for this workflow.
- `company_shift_share/tests/test_exposure_index.py`
  Synthetic tests for the new exposure-model logic.

## High-Level Flow

The pipeline has three layers.

### 1. Source Layer

Implemented in `source_exposure_data.py`.

This layer builds the shared raw objects that everything else depends on:

- preferred-RCID universe from the source preferred-RCID list,
- sampled outside negatives from Revelio company mapping,
- firm-year OPT counts from corrected FOIA,
- school-level OPT benchmarks from corrected FOIA,
- annual WRDS workforce cache for the full firm universe,
- annual WRDS school-flow cache for the full firm universe,
- and the final in-house firm-year analysis panel.

### 2. Feature Layer

Implemented in `revelio_company_features.py`.

This layer takes:

- the shared full firm universe,
- the WRDS annual caches,
- the FOIA school benchmark,
- and the FOIA firm-year OPT counts,

and turns them into a one-row-per-firm feature matrix with:

- static firm covariates,
- `*_pre_level`,
- `*_pre_growth`,
- `*_pre_n_years`,
- and level/growth missingness indicators.

### 3. Model / Event-Study Layer

Implemented in `exposure_event_study.py`.

This layer:

- loads the in-house analysis panel,
- loads the firm feature matrix,
- defines the post-2016 binary target,
- fits either an `lpm`, `lasso`, or `random_forest`,
- prints compact model / feature / predictive-power diagnostics in normal modeled-index runs,
- saves firm-level predictions,
- can instead run a testing-only diagnostic pass on a small FOIA-derived firm sample,
- and runs the event study using either:
  - legacy exposure measures from the new feature frame, or
  - the new modeled `opt_probability_index`.

## Data Sources

### Local parquet inputs

These are configured in `paths` in `configs/company_shift_share_exposure_event_study.yaml`.

- `foia_sevp_with_person_id_employment_corrected`
  Corrected FOIA F-1/OPT file.
- `employer_crosswalk`
  FOIA employer crosswalk used to map cleaned employer identifiers to preferred RCIDs.
- `preferred_rcids`
  Source preferred-RCID list.
- `revelio_company_mapping`
  Large Revelio company mapping parquet used for company metadata and outside-negative sampling.
- `f1_inst_unitid_crosswalk`
  FOIA school-name to `UNITID` crosswalk for building school OPT benchmarks.
- `revelio_ipeds_foia_inst_crosswalk`
  Legacy Revelio school crosswalk, used as one of the school-mapping sources.
- `revelio_inst_deterministic_map`
  Deterministic Revelio-school mapping file.
- `revelio_ref_inst_catalog`
  Institution catalog used with the deterministic school map.

### WRDS / Revelio tables

The WRDS queries are run directly against Revelio tables.

Current query surfaces include:

- `revelio.individual_positions`
- `revelio.individual_user`
- `revelio.individual_user_education`
- `revelio.individual_user_education_raw`

These queries are batched by RCID.

## What Is Built

Every persisted parquet also gets a sidecar metadata file:

- `*.parquet.meta.json`

Those metadata files are used to decide whether a cache can be reused.

### 1. `outside_negative_sample_out`

Default path:

- `data/out/company_shift_share_apr2026/outside_negative_sample.parquet`

Built in `load_or_build_source_firm_universe()` in `source_exposure_data.py`.

Contents:

- one row per sampled outside-negative firm,
- `c`
- `outside_negative_candidate = 1`

Sampling logic:

- candidate pool comes from `revelio_company_mapping`,
- must not be in the preferred-RCID list,
- must satisfy `n_users >= outside_negative_min_n_users`,
- must have usable state and NAICS,
- target count is `outside_negative_ratio × n_preferred_rcids`,
- sampling is stratified with fallback:
  - `size_bucket × naics2 × state`
  - `size_bucket × state`
  - `size_bucket`
  - global fill

Important:

- outside negatives are now included in the final analysis panel,
- not just in the exposure-model training frame.

### 2. `source_opt_counts_out`

Default path:

- `data/out/company_shift_share_apr2026/source_opt_counts.parquet`

Built in `build_source_opt_counts()` in `source_exposure_data.py`.

Contents:

- one row per `c × t`,
- firm-year OPT counts from corrected FOIA,
- degree-split counts:
  - `bachelors_opt_hires_correction_aware`
  - `masters_opt_hires_correction_aware`
  - `phd_opt_hires_correction_aware`
- pooled count:
  - `any_opt_hires_correction_aware`

Current method:

- corrected FOIA only,
- first spell per `person × employer × degree_group`,
- employer mapped to RCID through the source crosswalk,
- degree classified automatically into `bachelors`, `masters`, or `phd`.

### 3. `school_opt_benchmark_out`

Default path:

- `data/out/company_shift_share_apr2026/school_opt_benchmark.parquet`

Built in `build_source_school_opt_benchmark()` in `source_exposure_data.py`.

Contents:

- one row per `UNITID`,
- degree-specific average school OPT rates over the configured pre-period,
- degree-specific intensive-school labels:
  - `school_opt_rate_bachelors`
  - `school_opt_rate_masters`
  - `school_opt_rate_phd`
  - `opt_intensive_bachelors`
  - `opt_intensive_masters`
  - `opt_intensive_phd`

Construction:

- start from corrected FOIA,
- map `school_name` to `UNITID`,
- classify degree group,
- define valid OPT use based on employer presence and post-program start timing,
- compute school-year OPT rate,
- average within the configured pre-period,
- label schools above the within-degree median as OPT-intensive.

### 4. `wrds_company_year_workforce_out`

Default path:

- `data/out/company_shift_share_apr2026/wrds_company_year_workforce.parquet`

Built in `load_or_build_wrds_company_year_workforce_cache()` in `source_exposure_data.py`.

Contents:

- one row per `c × t` for the shared firm universe,
- annual workforce features from WRDS:
  - `total_headcount_wrds_annual`
  - `long_term_headcount_wrds_annual`
  - `salary_mean_annual`
  - `salary_var_annual`
  - `total_comp_mean_annual`
  - `total_comp_var_annual`
  - `compensation_missing_share_annual`
  - `nonus_educ_share_annual`
  - age-bin shares
  - `female_share_annual`
  - race-probability shares
  - `seniority_mean_annual`
  - `avg_tenure_years_annual`
  - occupation-group shares
  - `n_new_hires_wrds_annual`
- plus firm flags merged onto the cache:
  - `in_analysis_universe`
  - `preferred_rcid_source`
  - `outside_negative_candidate`

Important cache behavior:

- this cache is shared by both the analysis panel and the feature builder,
- so preferred firms are no longer queried twice.

### 5. `wrds_school_flows_out`

Default path:

- `data/out/company_shift_share_apr2026/wrds_school_flows.parquet`

Built in `load_or_build_wrds_school_flows_cache()` in `source_exposure_data.py`.

Contents:

- one row per `university_raw × c × t`,
- `n_transitions`
- `n_emp`
- `total_new_hires`

Interpretation:

- `n_transitions` is a first-job-after-graduation flow measure,
- `n_emp` is a long-term employee stock measure,
- both are later mapped to school IDs and combined with the FOIA school benchmark.

### 6. `opt_exposure_analysis_panel_out`

Default path:

- `data/out/company_shift_share_apr2026/opt_exposure_analysis_panel.parquet`

Built in `build_source_analysis_panel()` in `source_exposure_data.py`.

Contents:

- one row per `c × t` over the event-study data window,
- full firm universe, not just preferred-source firms,
- flags:
  - `in_analysis_universe`
  - `preferred_rcid_source`
  - `outside_negative_candidate`
- degree-split and pooled FOIA OPT counts,
- Revelio outcomes:
  - `y_cst_lag0`
  - `y_new_hires_lag0`

Important:

- `y_cst_lag0` is current-year WRDS headcount under the legacy compatibility name,
- `y_new_hires_lag0` is current-year WRDS new hires under the legacy compatibility name.

### 7. `company_features_out`

Default path:

- `data/out/company_shift_share_apr2026/company_features.parquet`

Built in `build_company_features()` in `revelio_company_features.py`.

Contents:

- one row per firm,
- static firm covariates:
  - `in_analysis_universe`
  - `preferred_rcid_source`
  - `outside_negative_candidate`
  - `naics2`
  - `naics4`
  - `company_state_feature`
  - `company_metro_feature`
  - `company_hq_region`
  - `company_age_feature`
  - `company_n_users_log1p`
- pre-period summaries for annual firm-year measures:
  - `*_pre_level`
  - `*_pre_growth`
  - `*_pre_n_years`
  - `*_pre_level_missing_ind`
  - `*_pre_growth_missing_ind`

Feature families currently included:

- pooled and degree-specific OPT hire counts,
- pooled and degree-specific OPT hire rates,
- pooled and degree-specific school OPT shares,
- number of schools contributing to new-hire and tenured-school measures,
- firm size and headcount,
- number of new hires,
- salary and total-comp mean / variance,
- compensation missing share,
- non-US education share,
- age shares,
- female share,
- race shares,
- mean seniority,
- average tenure,
- occupation-group shares.

Legacy compatibility:

- `opt_hire_count_annual` and `opt_hire_rate_annual` are master-based aliases,
- `school_opt_share_new_hire_annual` and `school_opt_share_tenured_annual` are master-based aliases.

### 8. `opt_probability_index_out`

Default path:

- `data/out/company_shift_share_apr2026/opt_probability_index.parquet`

Built in `_build_opt_probability_index()` in `exposure_event_study.py`.

Contents:

- one row per firm,
- firm-universe flags,
- `post2016_any_opt`
- `target_source`
- `leaveout_training_firm`
- `train_sample`
- `event_study_sample`
- `predicted_prob`
- `predicted_class`
- `predicted_index`
- `exposure_value`
- `model_method`
- `index_entry_mode`

Interpretation:

- for `lpm`, `predicted_index` is the clipped fitted value,
- for `lasso`, `predicted_index` is the fitted probability,
- for `random_forest`, `predicted_prob` is kept for diagnostics, but the event-study exposure is the hard `predicted_class`.

## Sample Definition

### Preferred-source firms

These come directly from `paths.preferred_rcids`.

If a `testing` section is enabled in the config, the preferred-source sample can be reduced to a smaller subsample before outside negatives are added.

Testing-mode preferred-source sampling is not arbitrary. The code:

- looks for preferred RCIDs that appear in source FOIA employer-linked counts during `testing.analysis_sample_year_min:analysis_sample_year_max`,
- tries to enforce minimum counts of post-2016 positives and non-positives using the configured target window,
- and then fills the remaining requested sample size with a seeded random draw.

### Outside negatives

These are sampled from `revelio_company_mapping` and tagged with:

- `preferred_rcid_source = 0`
- `outside_negative_candidate = 1`

### Final analysis universe

The final analysis panel and feature frame both use:

- `preferred_rcids ∪ outside_negatives`

The current code sets:

- `in_analysis_universe = 1` for both groups.

That means:

- outside negatives are part of the final event-study panel,
- not only part of model training.

## Config Structure

The recommended config is:

- `configs/company_shift_share_exposure_event_study.yaml`

It has four main sections.

### `paths`

Contains:

- raw local input files,
- and all persisted cache/output files.

The important output keys are:

- `source_opt_counts_out`
- `school_opt_benchmark_out`
- `outside_negative_sample_out`
- `wrds_company_year_workforce_out`
- `wrds_school_flows_out`
- `opt_exposure_analysis_panel_out`
- `company_features_out`
- `opt_probability_index_out`

When `testing.enabled = true`, the following outputs get `testing.output_suffix` appended to the filename so the sampled diagnostic run does not overwrite the full-sample outputs:

- `outside_negative_sample_out`
- `wrds_company_year_workforce_out`
- `wrds_school_flows_out`
- `opt_exposure_analysis_panel_out`
- `company_features_out`
- `opt_probability_index_out`

### `revelio_company_features`

Controls:

- WRDS username,
- pre-period feature window,
- outside-negative sampling,
- WRDS batching and timeout behavior,
- school-flow filters.

Important keys:

- `feature_year_min`
- `feature_year_max`
- `outside_negative_ratio`
- `outside_negative_seed`
- `outside_negative_min_n_users`
- `wrds_rcid_batch_size`
- `query_timeout_minutes`
- `query_max_retries`
- `min_position_days`
- `tenure_min_days`

### `exposure_event_study`

Controls:

- which exposure version to run,
- event-study windows,
- exposure-model settings,
- leave-out behavior,
- compact diagnostics for modeled-index runs,
- plotting/output behavior.

Important keys:

- `exposure_version`
  - `opt_hire_rate`
  - `school_opt_share`
  - `opt_probability_index`
  - `both`
  - `all`
- `exposure_year_min`
- `exposure_year_max`
- `feature_year_min`
- `feature_year_max`
- `target_year_min`
- `target_year_max`
- `index_model_method`
  - `lpm`
  - `lasso`
  - `random_forest`
- `index_entry_mode`
  - `ntiles`
  - `continuous`
- `leaveout_enabled`
- `leaveout_share`
- `leaveout_seed`
- `rf_n_estimators`
- `rf_max_depth`
- `rf_min_samples_leaf`
- `rf_min_samples_split`
- `force_rebuild_company_features`
- `force_rebuild_source_analysis_panel`
- `diagnostics_top_n`
- `predictive_n_bins`
- `ntiles`
- `event_year`
- `ref_year`
- `data_min_t`
- `data_max_t`
- `x_source_col`
- `outcome_cols`
- `slides_out_dir`
- `log_out_path`

### `testing`

Controls the sampled feature/model diagnostic mode used by `exposure_event_study.py`.

Important keys:

- `enabled`
- `verbose`
- `random_seed`
- `output_suffix`
- `analysis_sample_n`
- `analysis_sample_year_min`
- `analysis_sample_year_max`
- `analysis_min_post2016_positive`
- `analysis_min_post2016_nonpositive`
- `model_max_active_features`
- `model_max_feature_to_train_ratio`
- `feature_sample_seed`
- `diagnostics_top_n`
- `predictive_n_bins`

## SQL vs Pandas / DuckDB

The current workflow uses three execution environments.

### DuckDB

Used for local parquet work:

- FOIA school benchmark construction,
- FOIA OPT count construction,
- company-mapping scans for outside-negative sampling.

### WRDS SQL

Used for live Revelio pulls:

- annual workforce cache,
- annual school-flow cache.

These queries run remotely on WRDS and return pandas dataframes.

### Pandas

Used for:

- cache validation,
- merging blocks,
- sampling fallback logic,
- pre-period summarization,
- exposure-model fitting inputs,
- event-study regression setup.

## Exposure Model Details

### Target

The modeled target is:

- `post2016_any_opt = 1`

if a firm has any positive value of `x_source_col` in the target window.

In the minimal config:

- `x_source_col = any_opt_hires_correction_aware`
- `target_year_min = 2016`
- `target_year_max = 2022`

### LPM mode

- fit by OLS on the binary target,
- fitted values clipped to `[0, 1]`,
- event-study exposure can be:
  - `ntiles`, or
  - `continuous`

### LASSO mode

- uses `LogisticRegressionCV` with L1 penalty,
- internally cross-validates the penalty on the training sample,
- standardizes non-binary active features before fitting,
- event-study exposure can be:
  - `ntiles`, or
  - `continuous`

### Random forest mode

- uses `RandomForestClassifier`,
- event-study exposure is binary hard class,
- `continuous` is invalid,
- `ntiles` must be `2`

### Leave-out mode

If `leaveout_enabled = true`:

- the analysis-panel firms are split into a training subset and an event-study subset,
- the training draw is balanced across `preferred_rcid_source` and `outside_negative_candidate` when both sources are present,
- the preferred-source training draw is stratified on the post-2016 target when feasible,
- the split leaves firms from both sources in the event-study holdout when possible,
- firms used for model training are excluded from the event-study estimation sample.

If `leaveout_enabled = false`:

- the same analysis-panel firms are used for both fitting and event-study estimation.

### Diagnostics

For `opt_probability_index`, normal non-testing runs now emit a compact diagnostic summary before the outcome-by-outcome event-study passes:

- feature-family counts,
- a compact feature summary,
- model-weight summary,
- predictive-power summary,
- confusion matrix,
- and score-bin performance.

If testing mode is enabled, the script instead:

- bypasses all event-study estimation,
- runs the full feature-construction and model-fitting pipeline on the sampled firm universe,
- samples outside negatives from the local company-mapping universe without a WRDS US-position-count filter,
- prints either verbose or compact diagnostics depending on `testing.verbose`,
- includes feature, model-weight, and predictive-power summaries in both modes,
- and plots predictive-power diagnostics.

### Feature downsampling in testing mode

Testing samples can be small relative to the one-hot-expanded design matrix. The modeled index therefore supports feature downsampling before fitting:

- `testing.model_max_active_features` sets a hard cap on active design-matrix columns,
- `testing.model_max_feature_to_train_ratio` caps active columns as a multiple of the training-sample size,
- if either cap binds, the script keeps a seeded random subset of active features and reports that decision in diagnostics.

## How To Run

### Full pipeline

This is the main command:

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml
```

This will:

1. build or reuse the source firm universe,
2. build or reuse FOIA counts,
3. build or reuse school benchmark,
4. build or reuse WRDS workforce cache,
5. build or reuse WRDS school-flow cache,
6. build or reuse the source analysis panel,
7. build or reuse the company feature matrix,
8. fit the requested exposure model,
9. save predictions,
10. run the event study.

If `exposure_version = opt_probability_index`, this command also prints the compact modeled-index diagnostics described above.

### Build firm features only

```bash
python -m company_shift_share.revelio_company_features \
  --config configs/company_shift_share_exposure_event_study.yaml
```

Override the feature window:

```bash
python -m company_shift_share.revelio_company_features \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --feature-year-min 2012 \
  --feature-year-max 2015
```

Force rebuild the feature layer and all upstream caches it touches:

```bash
python -m company_shift_share.revelio_company_features \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --force-rebuild
```

### Run legacy exposure versions on the new source-built panel

`opt_hire_rate`:

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --exposure-version opt_hire_rate
```

`school_opt_share`:

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --exposure-version school_opt_share
```

All versions:

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --exposure-version all
```

### Run the modeled index in random-forest mode

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --exposure-version opt_probability_index \
  --index-model-method random_forest \
  --index-entry-mode ntiles \
  --ntiles 2
```

### Run the modeled index with leave-out enabled

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --exposure-version opt_probability_index \
  --leaveout-enabled \
  --leaveout-share 0.25 \
  --leaveout-seed 42
```

### Skip plot generation

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --no-plot
```

### Run testing mode

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --testing
```

This mode:

1. samples a small preferred-source firm subset from FOIA-linked firms,
2. adds sampled never-OPT outside negatives,
3. builds the full feature matrix and prediction file on that reduced universe,
4. skips all event-study regressions and outcome plots,
5. prints detailed feature / weight / predictive-power diagnostics,
6. and writes testing-suffixed caches and predictions.

### Write a run log

You can tee the full run output, including diagnostics and tracebacks, into a text-style file such as `.log`, `.txt`, or `.md`.

CLI override:

```bash
python -m company_shift_share.exposure_event_study \
  --config configs/company_shift_share_exposure_event_study.yaml \
  --testing \
  --log-file /tmp/exposure_event_study_testing.log
```

Config option:

```yaml
exposure_event_study:
  log_out_path: "{root}/data/out/company_shift_share_apr2026/exposure_event_study.log"
```

If testing mode is enabled, the configured log filename also gets `testing.output_suffix` appended so it does not overwrite the full-sample run log.

## Running Individual Source Builders

`source_exposure_data.py` is currently a library module, not a CLI script.

Use a Python entrypoint like this if you want to build individual source-layer objects:

```bash
python - <<'PY'
from company_shift_share.config_loader import load_config
from company_shift_share.source_exposure_data import (
    load_or_build_source_firm_universe,
    load_or_build_source_opt_counts,
    load_or_build_source_school_opt_benchmark,
    load_or_build_wrds_company_year_workforce_cache,
    load_or_build_wrds_school_flows_cache,
    load_or_build_source_analysis_panel,
)

cfg = load_config("configs/company_shift_share_exposure_event_study.yaml")

firms, outside, selected_meta, universe_meta = load_or_build_source_firm_universe(cfg=cfg, force_rebuild=False)
print("firm_universe", firms.shape, outside.shape, universe_meta)

opt_counts, opt_meta = load_or_build_source_opt_counts(cfg=cfg, year_min=2010, year_max=2022, force_rebuild=False)
print("source_opt_counts", opt_counts.shape, opt_meta)

school_bench, school_meta = load_or_build_source_school_opt_benchmark(cfg=cfg, year_min=2010, year_max=2015, force_rebuild=False)
print("school_benchmark", school_bench.shape, school_meta)

workforce, workforce_meta = load_or_build_wrds_company_year_workforce_cache(cfg=cfg, year_min=2010, year_max=2022, force_rebuild=False)
print("wrds_workforce", workforce.shape, workforce_meta)

flows, flow_meta = load_or_build_wrds_school_flows_cache(cfg=cfg, year_min=2010, year_max=2015, force_rebuild=False)
print("wrds_school_flows", flows.shape, flow_meta)

panel, panel_meta = load_or_build_source_analysis_panel(cfg=cfg, data_min_t=2010, data_max_t=2022, force_rebuild=False)
print("analysis_panel", panel.shape, panel_meta)
PY
```

## Forcing Rebuilds

### Feature CLI

`revelio_company_features.py` has a direct `--force-rebuild` flag.

### Event-study CLI

`exposure_event_study.py` does not currently have a top-level `--force-rebuild` flag.

To force rebuilds there, either:

- set these booleans in the YAML:
  - `exposure_event_study.force_rebuild_company_features: true`
  - `exposure_event_study.force_rebuild_source_analysis_panel: true`
- or remove the corresponding cached output files and rerun.

## Testing

### Fast syntax check

```bash
python -m py_compile \
  company_shift_share/source_exposure_data.py \
  company_shift_share/revelio_company_features.py \
  company_shift_share/exposure_event_study.py \
  company_shift_share/tests/test_exposure_index.py
```

### Unit tests

```bash
python -m unittest company_shift_share.tests.test_exposure_index
```

These tests cover:

- pre-period windowing,
- school benchmark windowing,
- RF validation rules,
- synthetic LPM / LASSO / RF fitting,
- interaction-feature construction,
- leave-out sample exclusion logic,
- testing-mode preferred-source sample selection,
- and active-feature downsampling.

### Testing mode in config

The shipped YAML now includes a `testing` section.

Supported keys:

- `enabled`
- `random_seed`
- `output_suffix`
- `analysis_sample_n`
- `analysis_sample_year_min`
- `analysis_sample_year_max`
- `analysis_min_post2016_positive`
- `analysis_min_post2016_nonpositive`
- `model_max_active_features`
- `model_max_feature_to_train_ratio`
- `feature_sample_seed`
- `diagnostics_top_n`
- `predictive_n_bins`

The main `exposure_event_study` section also supports:

- `log_out_path`

When enabled:

- the preferred-source firm sample is reduced before outside negatives are added,
- the model pipeline runs on that reduced universe,
- event-study estimation is skipped,
- verbose diagnostics are printed only if `testing.verbose: true`,
- otherwise a compact diagnostic summary is printed,
- and testing-suffixed outputs are written for the caches and predictions listed above.

Example:

```yaml
testing:
  enabled: true
  verbose: true
  random_seed: 42
  output_suffix: "_testing"
  analysis_sample_n: 60
  analysis_sample_year_min: 2010
  analysis_sample_year_max: 2015
  analysis_min_post2016_positive: 15
  analysis_min_post2016_nonpositive: 15
  model_max_active_features: null
  model_max_feature_to_train_ratio: 0.75
  feature_sample_seed: 42
  diagnostics_top_n: 25
  predictive_n_bins: 8
```

## Current Caveats

- `source_exposure_data.py` does not yet have its own CLI wrapper.
- The event-study CLI does not yet expose every internal model-tuning argument as command-line flags.
- The unit tests are synthetic and do not hit WRDS.
- Full runtime and query cost still depend on the live WRDS environment and the chosen firm universe.

## Legacy Note

This folder still contains the older company shift-share instrument builders, including `build_company_shift_share.py`.

Those legacy scripts:

- are still available,
- but are not upstream of the source-only OPT exposure pipeline documented here.
