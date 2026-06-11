# relabel_events_generalized Reference Build Spec

This document summarizes the current working-tree implementation of
`code/relabels_revelio/relabel_events_generalized.py`.

It is meant as a reference for understanding the legacy pipeline. It is not a
clean-room specification for `cip_relabel_jmp`; do not copy implementation
details into the clean-room repo unless the step has been independently
specified there.

## Entry Point

Run the legacy pipeline with:

```bash
python code/relabels_revelio/relabel_events_generalized.py
```

The script calls `run_pipeline(...)`, which returns three data frames:

- `events`: merged event table
- `panel`: verified event panel
- `candidate_audit`: external-candidate audit table

## Default Inputs

- IPEDS completions: `base.IPEDS_PATH`
- IPEDS cost panel: `base.IPEDS_COST_PANEL_PATH`
- STEM OPT CIP long file: `base.STEM_OPT_LONG_PATH`
- External candidate path: `${base.root}/data/llm_relabel_candidates`
- IPEDS crosswalk: `v2.CROSSWALK_PATH`
- FOIA file: `base.FOIA_PATH`
- FOIA institution crosswalk: `base.F1_INST_CW_PATH`
- FOIA person panel: `base.FOIA_PERSON_PANEL_PATH`
- Employer match directory:
  `${base.root}/data/company_matching_f1_apr2026`
- IPEDS main institutions:
  `${base.root}/data/int/int_files_feb2026/ipeds_main_institutions.parquet`
- IPEDS 2024 directory:
  `${base.root}/data/raw/ipeds/directory_info_hd/hd2024.csv`

## Default Outputs

All default outputs are under:

```text
${base.root}/h1bworkers/code/output/relabel_indiv/
```

- `generalized_relabels_events.parquet`
- `generalized_relabels_events.csv`
- `generalized_relabels_panel.parquet`
- `generalized_relabels_report.txt`
- `generalized_relabels_candidate_audit.csv`
- `generalized_relabels_plots/`

## Global Defaults

- Relabel year mode: `first`
- Event-study estimator: `did`
- DiD spec: `individual_broad_bin_degree_fe`
- Control group: `never_treated`
- Event-time plot window: `[-5, 4]`
- DiD reference event time: `-2`
- FOIA outcome max year: `min(base.PLOT_YEAR_MAX, 2022)`
- IPEDS outcome max year: `2024`
- Max relabel year: `2021`
- Broad treated event minimum relabel year: `2014`
- Degree-specific plot bundles: disabled by default

## Broad CIP Bins

The legacy pipeline maps exact CIP pairs into broad source-to-target bins.

| Broad bin | Source CIPs | Target CIPs |
| --- | --- | --- |
| `econ_to_quant_econ` | prefix `4506`, excluding exact `450603` | exact `450603` |
| `business_52_to_52` | prefix `52`, excluding prefixes `5208`, `5213` | prefix `5213` |
| `finance_to_quantitative_finance` | prefix `5208` | exact `270305`, `270501` |
| `communication_to_digital_media` | prefixes `0901`, `0904`, `0909`, excluding exact `090702` | exact `090702` |
| `architecture_design_to_built_env_stem` | prefixes `0402`, `0403`, `0406`, `5004` | exact `040902`, `303301` |
| `agricultural_econ_to_mathematical_econ` | exact `010103` | exact `304901` |

`communication_to_digital_media` is computed as a broad bin but excluded from
candidate rows and event rows before analysis.

## Step 01: Load and Normalize External Candidates

Function: `load_external_candidates(...)`

Inputs:

- A CSV, Excel, parquet file, or directory containing supported files.

Rules:

- Infer required columns for school, approximate year, program description, and
  degree label using permissive header aliases.
- Infer optional candidate ID, notes, source CIP, and target CIP columns when
  present.
- Canonicalize common school names, including overrides such as `ucla`, `mit`,
  `uc berkeley`, `usc`, `ut austin`, and related aliases.
- Extract candidate year from the year/date field.
- Normalize degree type to `Bachelor`, `Master`, `Doctor`, or `Other`.
- Build a program signature and candidate major from the program description.
- Extract source and target CIP6 hints where present.
- Parse candidate CIP constraints into source bin, target bin, pair bin, and
  parse notes.
- Generate candidate IDs from file stem and row number when IDs are missing.
- For directory inputs, concatenate all supported files and drop hard-coded
  known duplicate LLM rows.
- Exclude candidates whose broad bin maps to
  `communication_to_digital_media`.

Output:

- Cleaned candidate frame used for school resolution, verification, and audit.

## Step 02: Derive Allowable CIP Pair Configurations

Function: `derive_allowable_pair_configs(...)`

Purpose:

- Restrict the strict IPEDS scan to source-target CIP patterns implied by the
  cleaned external candidates when candidates are supplied.

Rules:

- Parse each candidate row into source and target CIP rules.
- Deduplicate rule pairs by rule signature.
- If no candidate input is provided, the strict IPEDS scan is unrestricted.

Output:

- List of allowable source-target CIP rule configurations.

## Step 03: Strict IPEDS Broad-Bin Relabel Scan

Functions:

- `detect_ipeds_relabels(...)`
- `scan_ipeds_broad_bin_candidates(...)`
- `scan_ipeds_pair_candidates(...)`

Inputs:

- IPEDS completions parquet.
- Strict thresholds.
- Optional allowable source-target CIP configurations.

Strict thresholds:

- Minimum international share: `0.20`
- Minimum source baseline: `10`
- Minimum source drop count: `5`
- Minimum source drop percent: `0.50`
- Minimum target offset share: `0.60`
- Maximum net loss share: `0.50`
- Source persistence drop share: `0.30`
- Target persistence gain share: `0.30`
- Lookback years: `3`
- Lookahead years: `2`

Rules:

- Stage IPEDS as a DuckDB view.
- Restrict to rows with nonmissing `unitid`, `cipcode`, and `awlevel`.
- Keep candidate rows where `share_intl` is at least the threshold.
- Build a complete unit-awlevel-CIP-year panel over the IPEDS year range.
- Detect source drops using lagged totals and pre/post window averages.
- Pair each source drop with target CIP growth in the same unit, awlevel, and
  year.
- Score pairs using target offset, persistent target growth, persistent source
  drop, and net-loss penalties.
- Keep the best source pair and, unless `keep_all_sources=True`, the best
  unit-awlevel-year pair.
- Convert exact source-target CIP pairs to broad bins.
- Apply the master/doctorate guard for relevant master events.
- Collapse repeated same unit-awlevel-source-target events by either earliest
  year (`first`) or highest score (`largest`).
- Rename `source_cip6` to `event_source_cip6`.
- Mark these rows as `found_in_ipeds_scan=1`,
  `found_in_external_candidates=0`, `external_verified=0`,
  `event_origin_category=ipeds_only`.
- Exclude disallowed broad bins.

Output:

- Strict IPEDS event frame.

## Step 04: Resolve Candidate Schools

Functions:

- `load_school_lookup(...)`
- `resolve_candidate_schools(...)`

Inputs:

- Cleaned candidates.
- IPEDS crosswalk.

Rules:

- Load school names and unit IDs from the crosswalk.
- Normalize school names.
- Use explicit candidate-name overrides where available.
- Match candidates to schools using exact or fuzzy name matching.
- Record match method, score, matched name, and matched `unitid`.

Output:

- Candidate frame with `matched_unitid`, school match metadata, and original
  candidate fields.

## Step 05: Verify External Candidates Against IPEDS

Function: `verify_external_candidates(...)`

Inputs:

- Resolved candidate rows.
- IPEDS completions.
- CIP map.
- Verification thresholds.

Rules:

- For each candidate, map degree type to IPEDS awlevels:
  `Bachelor -> 5`, `Master -> 7`, `Doctor -> 9 or 17`.
- Search within `approx_year - 3` through `approx_year + 3`, clamped to the
  analysis relabel-year window.
- Skip with audit notes if the school is unmatched, degree is unmappable, or
  candidate CIP bins cannot be parsed.
- Run a nearby IPEDS pair scan using the verification thresholds.
- Filter nearby pairs to those satisfying the candidate source and target CIP
  rules.
- Rank matches by relabel score plus a year-distance bonus and text-similarity
  bonus against source/target CIP labels.
- If no qualifying pair is found, run a diagnostic scan with permissive
  thresholds and record the best diagnostic match.
- If a match is found, emit a verified event row and mark the audit row
  `external_verified=1`.

Current implementation note:

- `run_pipeline(...)` passes `STRICT_THRESHOLDS` into
  `verify_external_candidates(...)`, even though the function default is named
  `RELAXED_THRESHOLDS`.

Outputs:

- `verified_external`: externally supplied rows verified by IPEDS.
- `candidate_audit`: all candidate rows with verification status, best-match
  metadata, and diagnostic metadata.

## Step 06: Merge Event Sources

Function: `merge_event_sources(...)`

Inputs:

- Strict IPEDS events.
- Verified external events.
- Candidate audit rows.
- CIP labels.

Rules:

- Define event key as:
  `(unitid, awlevel, relabel_year, event_source_cip6, target_cip6)`.
- Merge strict and externally verified rows by event key.
- Preserve positive provenance flags across duplicate event keys.
- Aggregate linked candidate metadata for verified candidates sharing an event
  key.
- Add unverified candidates as `external_only` rows with no event flag.
- Fill CIP labels and cleaned major labels.
- Set `event_flag=1` and `relabel_flag=1` for verified IPEDS-backed rows;
  set both to `0` for `external_only`.
- Annotate broad bins and broad-bin eligibility.
- Keep and order columns according to `VERIFIED_EVENT_COLUMNS`.
- Sort by degree type, event origin, relabel year, unit ID, source CIP, and
  target CIP.
- Exclude disallowed broad bins.
- Filter to the relabel analysis window.
- Coerce output dtypes for string, integer, and float event columns.

Output:

- `generalized_relabels_events.parquet`
- `generalized_relabels_events.csv`

## Step 07: Build Broad Treated Events

Function: `build_broad_treated_events(...)`

Inputs:

- Merged event table.

Rules:

- Keep only `ipeds_only` and `external_ipeds_verified` rows.
- Require nonmissing relabel year, awlevel, and broad-bin eligibility.
- Keep relabel years between `2014` and `2021`.
- Collapse to one treated event per `(unitid, awlevel, broad_pair_bin)`.
- If relabel-year mode is `first`, choose the earliest relabel year, breaking
  ties by higher relabel score.
- If relabel-year mode is `largest`, choose the highest relabel score, breaking
  ties by later relabel year.
- Set `relabel_type` to the broad pair bin and `year` to `relabel_year`.
- Recompute provenance flags at the grouped-event level.

Output:

- Broad-bin treated events used to build the event panel.

## Step 08: Build Verified Event Panel

Function: `build_verified_event_panel(...)`

Inputs:

- Merged events.
- IPEDS completions.
- Relabel year mode.

Rules:

- Build broad-bin membership over the IPEDS CIP universe.
- For each broad treated event, collect all source and target CIPs belonging to
  the event's broad bin.
- Build annual panel rows from the minimum IPEDS year through
  `ANALYSIS_IPEDS_YEAR_MAX` (`2024`), bounded by available IPEDS data.
- Aggregate source-side and target-side `ctotalt` and `cnralt` by year.
- Populate current and lagged source/target totals.
- Set `ctotalt = source_total + target_total`.
- Set `cnralt = source cnralt + target cnralt`.
- Set `event_flag=1` only in the relabel year.
- Carry event metadata and candidate/audit fields onto every year row.
- Sort by degree type, broad bin, unit ID, relabel year, and year.
- Coerce output dtypes.

Output:

- `generalized_relabels_panel.parquet`

## Step 09: Stage FOIA and Employer-History Inputs

Functions:

- `_load_foia_base(...)`
- `_stage_foia_employer_history_views(...)`
- `_stage_foia_analysis_base(...)`

Inputs:

- FOIA records.
- FOIA institution crosswalk.
- FOIA person panel.
- Employer entity-to-firm artifacts.
- STEM OPT CIP first-year file.

Rules:

- Read FOIA and crosswalks into DuckDB.
- Resolve FOIA schema using the v2 helper.
- Add a zero-based `original_row_num` to FOIA rows.
- Stage FOIA person-employer history and row-to-firm mappings.
- Normalize employer names, city, state, and ZIP for firm lookup.
- Construct person/spell identifiers, including fallback spell keys.
- Build employer OPT-intensity percentiles from distinct firm-person counts.
- Build `foia_analysis_base` with:
  `unitid`, normalized `cip6`, graduation year, student ID, STEM eligibility,
  employer fields, OPT type and dates, tuition/funding fields, program end
  date, requested status, and FOIA degree label.
- Restrict FOIA records to nonmissing program end date, matched institution,
  nonmissing CIP, nonmissing graduation year, and graduation year no later than
  the FOIA analysis max year.

Output:

- DuckDB temp tables used by outcome and DiD construction.

## Step 10: Compute Treated FOIA/IPEDS Outcomes

Function: `compute_opt_usage_generalized(...)`

Inputs:

- Verified event panel.
- Degree type.
- FOIA, institution crosswalk, person panel, employer match artifacts.
- IPEDS cost panel.

Rules:

- Keep panel rows for the selected degree type, verified origins, and
  broad-bin eligible rows.
- Join FOIA rows to event rows on unit ID, graduation/calendar year, and FOIA
  degree label.
- Keep FOIA CIPs belonging to the event broad bin.
- Aggregate to student level, then calendar-year/relabel-type/degree cells.
- Count total graduates and users with STEM CIP eligibility, OPT, STEM OPT, and
  status change.
- Compute post-graduation authorization years and OPT duration years.
- Compute employer-history outcomes: unique employers, unique OPT cities,
  authorization employment tenure, employer OPT intensity percentile,
  internship count, and internship OPT years.
- Attach IPEDS `ctotalt`, `cnralt`, tuition, and fees.
- Compute rates and averages using safe division.

Main output variables:

- `stem_cip_eligible_share`
- `opt_share`
- `opt_stem_share`
- `status_change_share`
- `post_grad_authorization_years_avg`
- `opt_duration_years_avg`
- `unique_employers`
- `unique_opt_cities`
- `auth_employment_tenure_years`
- `employer_opt_intensity_pctile`
- `internship_count`
- `internship_opt_years`
- `ctotalt`
- `cnralt`
- `cnralt_share_of_ctotalt`
- `f1_share_of_ctotalt`
- `f1_share_of_cnralt`
- `avg_tuition`
- `avg_tuition_ipeds`
- `avg_fees_ipeds`
- `avg_students_personal_funds`
- `avg_total_funds`

## Step 11: Convert Treated Outcomes to Event Time

Function: `compute_opt_usage_event_time_generalized(...)`

Rules:

- Filter to the relabel analysis window.
- Define `event_t = calendar_year - relabel_year`.
- Keep event times from `-5` through `4`.
- Group by `event_t`, `relabel_type`, and `degree_type`.
- Sum count and total variables.
- Recompute rates and averages using safe division.

Output:

- Event-time treated outcome table for raw event-time plotting.

## Step 12: Construct Matched Controls

Functions:

- `match_treated_to_never_treated(...)`
- `_late_treated_control_events(...)`
- `_alternate_control_candidate_pool(...)`
- `_always_stem_cip_rows(...)`

Supported control groups:

- `never_treated`
- `late_treated`
- `always_stem`

General rules:

- Start from treated broad-bin events.
- Summarize treated pre-period source totals over the lookback window.
- Build candidate control pools from IPEDS programs/institutions that are not
  already treated in the same broad bin and degree group.
- Match on pre-period source level and pre-period growth.
- Apply a size-ratio caliper when the treated pre-period size is at least `5`:
  controls must be between `0.25x` and `4.0x` the treated size.
- Attach Carnegie basic classification where IPEDS columns are available.
- Prefer no replacement; allow replacement when needed.

Control-group-specific rules:

- `never_treated`: controls are institutions/programs in the same broad bin
  that do not become treated in the treated analysis window.
- `late_treated`: controls are future treated events from relabel years
  `2020` through `2022`, and must occur after the treated event year.
- `always_stem`: controls are always-STEM CIPs whose first STEM year is no
  later than `2014`, limited to CIP2 families `11`, `14`, `26`, and `40`.

Output:

- Matched pair table with treated unit, control unit, relabel year, broad bin,
  degree type, match distance, pre-period summaries, control metadata, and a
  replacement flag.

## Step 13: Build Matched FOIA DiD Panel

Function: `compute_generalized_did_panel(...)`

Inputs:

- Verified event panel.
- Degree type or pooled degree mode.
- DiD spec.
- Control group.
- FOIA/IPEDS/employer inputs.

Rules:

- Keep selected degree(s), broad-bin eligible rows, and analysis-window events.
- Match treated events to controls.
- Build treated and control outcome rows using the same FOIA, IPEDS, STEM,
  cost, and employer-history logic.
- For treated rows, use all CIPs in the broad bin.
- For control rows, use matched control CIPs:
  all broad-bin CIPs for standard controls, selected always-STEM CIPs for the
  always-STEM control group.
- Concatenate treated and control rows and set `treated` indicator.
- Define `event_t = calendar_year - relabel_year`.
- Recompute shares, averages, tuition/funding measures, and IPEDS ratios.
- Apply DiD design columns.
- Set `panel_level` to `individual` for
  `individual_broad_bin_degree_fe`; otherwise `collapsed`.

DiD specs:

- `collapsed_unit_fe`: aggregated school-year panel.
- `individual_broad_bin_degree_fe`: individual rows with unitid-by-broad-bin
  by degree fixed effects where available.

Output:

- Matched FOIA DiD panel used for raw means, post summaries, and event-study
  regressions.

## Step 14: Build Matched IPEDS DiD Panel

Function: `compute_generalized_ipeds_did_panel(...)`

Inputs:

- Verified event panel.
- Degree type or pooled degree mode.
- DiD spec.
- Control group.
- IPEDS completions.

Rules:

- Match treated events to controls using the same matching routine.
- Build a treated/control pair-role year grid from the original analysis year
  minimum through the IPEDS analysis max year, bounded by available IPEDS data.
- Aggregate `ctotalt` and `cnralt` over treated broad-bin CIPs and matched
  control CIPs.
- Define `event_t`.
- Compute `cnralt_share_of_ctotalt`.
- Apply DiD design columns.
- Set `panel_level = ipeds_program`.

Output:

- Matched IPEDS DiD panel for IPEDS-only outcomes.

## Step 15: Estimate Event Studies and Post Summaries

Functions:

- `compute_did_event_study_generalized(...)`
- `compute_stacked_event_study_generalized(...)`
- `compute_post_did_summary(...)`
- `build_did_summary_text(...)`

Rules:

- Restrict regressions to event times `[-5, 4]`.
- Drop rows missing the outcome, event time, treatment indicator, unit ID,
  calendar year, or weights.
- Require common event-time support among treated and controls.
- Use reference event time `-2`.
- For standard DiD, estimate event-time-by-treated coefficients with fixed
  effects determined by the DiD spec.
- For stacked treated-only event study, exclude controls and estimate
  event-time coefficients relative to the reference period.
- Use pyfixest where available for the configured design, with statsmodels
  fallbacks in helper routines.
- Post summary treats `event_t >= -1` as post and uses the pre-period through
  the reference event time as baseline.

Output:

- Event-study coefficient frames with `event_t`, `coef`, `se`, confidence
  interval fields, counts, and reference-event metadata.

## Step 16: Generate Plots and Appendices

Function: `run_degree_plots(...)`

Always attempted:

- Broad-bin event counts by year.
- Broad-bin degree-year breakdown.
- Sample treated/control schools by broad bin.
- Source/target `ctotalt` event-time plot by treated/control and degree.
- Pooled FOIA matched DiD panel.
- Pooled IPEDS matched DiD panel.
- Pooled raw means and DiD event studies for configured outcomes.
- Grouped DiD appendix plots.

If `--include-degree-specific-plots` is supplied:

- Degree-specific relabel-year histograms.
- Degree-specific treated outcome calendar plots.
- Degree-specific treated/control raw event-time plots.
- Degree-specific DiD event-study plots.
- Degree-specific stacked treated-only plots when requested.

Estimator options:

- `did`: standard treated-vs-control DiD plots.
- `stacked_treated`: treated-only stacked event-study plots.
- `both`: both estimator families.

Output:

- PNG plot files and appendix markdown files under
  `generalized_relabels_plots/`.

## Step 17: Write Report

Function: `write_generalized_report(...)`

Report contents:

- IPEDS and FOIA source paths.
- Candidate audit count.
- Candidate verification summary.
- Event counts by origin.
- Plot appendix paths.
- Event counts by degree and origin.
- Source and target major counts by origin.
- Candidate major counts for unverified external rows.
- Candidate audit appendix with candidate ID, school, year, degree, program,
  parsed bins, school match, verification status, best nearby match, and notes.

Output:

- `generalized_relabels_report.txt`

## CLI Options

- `--candidate-path`
- `--ipeds-path`
- `--crosswalk-path`
- `--foia-path`
- `--inst-cw-path`
- `--foia-person-panel-path`
- `--employer-match-dir`
- `--events-parquet`
- `--events-csv`
- `--panel-parquet`
- `--report-path`
- `--candidate-audit-csv`
- `--plots-dir`
- `--did-spec {collapsed_unit_fe,individual_broad_bin_degree_fe}`
- `--estimator {did,stacked_treated,both}`
- `--relabel-year-mode {first,largest}`
- `--include-degree-specific-plots`

## Suggested Comparison Artifacts for Clean-Room Work

Use these legacy outputs as comparison targets, not as implementation inputs:

- Events:
  `code/output/relabel_indiv/generalized_relabels_events.parquet`
- Events CSV:
  `code/output/relabel_indiv/generalized_relabels_events.csv`
- Verified panel:
  `code/output/relabel_indiv/generalized_relabels_panel.parquet`
- Candidate audit:
  `code/output/relabel_indiv/generalized_relabels_candidate_audit.csv`
- Report:
  `code/output/relabel_indiv/generalized_relabels_report.txt`

Recommended step-level checks:

- Candidate row counts by source file, degree type, school match method, and
  verification note.
- Event row counts by origin, degree type, broad bin, relabel year, and
  unit-awlevel-broad-bin key.
- Verified panel key uniqueness over unit ID, awlevel, broad bin, relabel year,
  and year.
- Source/target annual totals for selected known events.
- Matched-pair counts and match-distance summaries by control group.
- DiD panel counts by treated status, degree type, broad bin, event time, and
  panel level.
- Outcome means and denominators by event time for each plotted outcome.
