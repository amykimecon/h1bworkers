# Source-Rebuilt Matched `z_ct` Design

## Purpose
This design rebuilds the firm-year exposure panel from source inputs rather than
reading any saved final analysis panel from the older shift-share workflow.

The pipeline:

1. rebuilds school shocks with the existing persistent-shock methodology from
   `shift_share_analysis.py`,
2. rebuilds firm-school transition shares from source-side WRDS school flows for
   both preferred firms and outside negatives,
3. reconstructs baseline `z_ct = sum_k share_ck * g_kt`,
4. groups firms by their full `z_ct` path,
5. estimates an OPT-takeup propensity score on non-school, non-OPT firm
   covariates while keeping `nonus_educ_share_*`,
6. matches high-exposure preferred firms to low-exposure controls, and
7. runs a matched common-break event study plus a matched stacked DiD.

## Data Lineage
- School shocks come from the same raw IPEDS and FOIA inputs used by
  `shift_share_analysis.py`.
- Firm-school shares come from `load_or_build_wrds_school_flows_cache`, not from
  the saved `revelio_school_to_employer_transitions.parquet` output used in the
  older shift-share panel build.
- Outcomes come from the source-side OPT counts plus WRDS workforce caches.
- Matching covariates come from static company metadata plus pre-period WRDS
  workforce summaries. The design does not depend on any saved final
  `company_features.parquet` output.

## Shift-Share Rebuild
- School-side defaults mirror the April 2026 shift-share baseline:
  `share_period=pre_window`, share window `2008-2013`, event window
  `2014-2017`, `event_shock_pre_years=2`, `event_shock_post_years=2`, and the
  persistent step shock used as baseline `g_kt`.
- The matched-design config can override `school_sample_window_start`,
  `school_sample_window_end`, `event_shock_pre_years`, and
  `event_shock_post_years` without editing the base shift-share config.
- The current testing config also overrides the share window to `2010-2013` so
  the design can be exercised without expanding WRDS school-flow coverage back
  to 2008. Revert to `2008-2013` for the intended production baseline.
- The matched-school sampler is reused so the school-side shock sample stays
  aligned with the baseline persistent-shock design.
- Source-side firm-school shares are rebuilt by mapping `university_raw` to
  IPEDS `unitid` via `load_revelio_school_map`, then applying the existing
  share-window formulas from `shift_share_analysis._build_transition_shares`.
- Missing firm-year exposure rows are filled with zero after the firm-year
  cross join so every firm has a defined `z_ct` path.

## Trajectory Grouping
- Main grouping window: `2010-2015`.
- Robustness grouping window: `2010-2022`.
- For each window, compute:
  - `cum_z = sum_t z_ct`
  - `max_z = max_t z_ct`
  - `mean_z = mean_t z_ct`
  - `active_share = mean(1[z_ct > 0])`
- High exposure:
  preferred-source firms with the configured high-exposure metric at or above
  the 75th percentile among preferred firms with positive exposure on that
  metric. The default metric is `max_z`.
- Low exposure:
  firms with the configured low-exposure metric equal to zero, or at or below
  the 25th percentile among preferred firms with positive exposure on that
  metric, excluding firms already assigned to high exposure. The default metric
  is `max_z`.
- Middle-group firms are dropped from the matched design.
- Outside negatives can enter the control pool only if
  `include_outside_negative_controls=true` and their rebuilt trajectory is
  classified as low exposure.
- `active_share` is still computed for diagnostics, but it is not part of the
  default exposure classification rule.

## Propensity Score
- Target:
  `post2016_any_opt` over `2016-2022`, matching the source exposure-event-study
  target.
- Default model:
  unpenalized logit with balanced class weights.
- Default training sample:
  preferred firms plus outside negatives coded as zero.
- Optional robustness mode:
  preferred-only training while still scoring all firms.
- Feature family:
  exclude every `school_*`, `n_schools_*`, and `*_opt_*` column, but keep
  industry, geography, age, size/headcount, hiring, compensation, workforce
  composition, tenure, seniority, occupation mix, and `nonus_educ_share_*`.
- `leaveout_enabled` is configurable and defaults to `false` in this design.
- The reporting layer writes the propensity-model construction details to
  `tables/propensity_model_diagnostics.csv`, including training counts, raw and
  active feature counts, class weighting, interaction-feature counts when
  applicable, and holdout/evaluation AUC and Brier statistics.
- The fitted design-matrix weights are written to
  `tables/propensity_feature_weights.csv`, ranked by absolute coefficient or
  feature importance. For logit/lasso, coefficients are on the model matrix
  after median imputation, dummy expansion, and standardization of non-binary
  design columns. `tables/propensity_feature_group_weights.csv` also aggregates
  those design-column weights back to the source covariate where possible, so a
  categorical covariate's dummy columns can be read together.
- The score distribution by trajectory, exposure group, and firm source is
  written to `tables/propensity_score_distribution.csv`.

## Matching
- Match each high-exposure preferred firm to one low-exposure control.
- Distance metric:
  nearest neighbor on the logit propensity score, with an explicit size/growth penalty:
  - pre-period firm size (`match_firm_size_pre_level`) and size-growth
    (`match_firm_size_pre_growth`) are added to the matching feature set.
  - the matcher minimizes:

`|logit_score_i - logit_score_j| + w_size × |size_i - size_j|/sd(size) + w_growth × |growth_i - growth_j|/sd(growth)`

  with defaults `w_size=0.35`, `w_growth=0.35` (set via
  `matching_size_weight` and `matching_growth_weight`).
- Restrictions:
  exact `naics2`, pooled common-support trimming, and a default caliper of
  `0.2 × SD(logit score)`.
- Matching is without replacement.
- Region compatibility is a secondary tie-breaker after score/size/growth distance.
- Progress logging:
  the matcher prints a start line, a search-space summary, periodic progress
  updates every `matching_progress_every` treated firms, and a final completion
  line with unmatched reasons and elapsed time.
- Manual reuse mode:
  if `reuse_saved_analysis_outputs=true` and the saved matched-design artifacts
  already exist, the script loads those outputs from disk and rewrites the
  reporting layer without rerunning the heavy source build, matching, and
  regression stages. This is intended for report/diagnostic changes, not for
  substantive design changes.

## Regressions
- Common-break event study:
  - matched sample only; model:

`y_{icpt} = Σ_{k≠ref_year} β_k·1[t=k]·treated_i + α_i + λ_t + ε_{icpt}`

  - `treated_i=1` for high-exposure matched firms, zero otherwise
  - firm (`α_i`) and year (`λ_t`) fixed effects
  - standard errors clustered by `pair_id`
  - coefficients are estimated for each outcome and trajectory, including the omitted
    reference year `ref_year` (default 2014)
- Stacked DiD:
  - treated cohort year `g_c` is the first year with positive `z_ct` satisfying
    the stack rules, then each matched pair is stacked by `rel_time = t - g_c`
  - controls are retained only if exposure stays zero for all stack window years
  - model:

`y_{icpt} = Σ_{k≠-1} β_k·1[rel_time=k]·treated_i + η_{pair×stack} + δ_{year×stack} + u`

  - controls for stack-specific pair and year fixed effects
  - standard errors clustered by original `pair_id`

## Outputs
The pipeline writes:

- rebuilt transition shares,
- rebuilt school growth panel,
- rebuilt instrument panel,
- rebuilt firm-year analysis panel,
- matching feature frame,
- propensity-score predictions,
- trajectory summaries,
- matched pairs,
- matching balance table with pre/post-match balance rows for overall and
  control-source-specific comparisons,
- matching balance summary with max/mean absolute SMD and threshold counts by
  stage and control source,
- full detailed balance CSV at `tables/matching_balance_all_covariates.csv`,
  with every numeric covariate and every categorical indicator row used in the
  balance calculation,
- matching diagnostics with elapsed time, search-space size, and average
  candidate-pool sizes before and after the caliper,
- CSV analysis tables under `out_dir/tables/`, including trajectory counts,
  propensity-score summaries, propensity model diagnostics, feature-weight
  tables, matching diagnostics, detailed balance rows, top balance imbalances,
  long plus summary tables for the common-break and stacked-DiD results,
  plus treated-control raw mean tables for both event studies
- PNG analysis figures under `out_dir/figures/`, including balance-summary
  plots, one coefficient plot per trajectory/outcome for both the common-break
  event study and stacked DiD, plus treated-control raw outcome mean trajectories for
  both event studies
- a markdown run summary at `out_dir/analysis_summary.md`,
- matched firm-year panel,
- final matched firm-year panel artifact (same panel written after all trajectory-level matching),
- common-break event-study coefficients,
- stacked-DiD panel,
- stacked-DiD coefficients, and
- a diagnostics JSON file.

## Methodology References
- [Heckman, Ichimura, and Todd 1997](https://academic.oup.com/restud/article/64/4/605/1603767)
- [Heckman, Ichimura, and Todd 1998](https://academic.oup.com/restud/article/65/2/261/1580756)
- [Dehejia & Wahba](https://www.nber.org/papers/w6829)
- [Smith & Todd 2005](https://www.sciencedirect.com/science/article/pii/S030440760400082X)
- [Abadie 2005](https://academic.oup.com/restud/article/72/1/1/1581053)
- [Abadie & Imbens 2006](https://www.econometricsociety.org/publications/econometrica/2006/01/01/large-sample-properties-matching-estimators-average-treatment)
- [Abadie & Imbens 2011](https://www.hks.harvard.edu/publications/bias-corrected-matching-estimators-average-treatment-effects)
- [Callaway & Sant’Anna 2021](https://www.sciencedirect.com/science/article/pii/S0304407620303948)
- [Sun & Abraham 2021](https://www.sciencedirect.com/science/article/pii/S030440762030378X)
- [Rosenbaum & Rubin 1983](https://www.jstor.org/stable/2335942)
- [Stuart 2010](https://psycnet.apa.org/record/2010-19790-010)

The matching stage is treated as design preprocessing. Regression inference is
pair-clustered. The note does not treat naive bootstrap inference as valid for a
standalone nearest-neighbor matching estimand.

## Balance Diagnostics
- The balance table is built on the raw matching covariates used in the
  propensity-score design, plus `predicted_prob` and the logit score used for
  nearest-neighbor matching.
- Numeric covariates are reported with treated/control means, mean differences,
  pooled-SD standardized mean differences, variance ratios, and a normal
  approximation p-value for the mean-difference test.
- Categorical covariates are expanded to indicator rows of the form
  `column=value`, and the same balance statistics are reported on the
  indicator shares.
- For each trajectory spec, the pipeline writes:
  - `pre_match` balance for the common-support eligible pool,
  - `post_match` balance for the realized matched sample,
  - `overall` balance,
  - `preferred_low_exposure` control-only balance, and
  - `outside_negative` control-only balance when outside negatives are used in
    the eligible or matched pool.
- The full long balance table is saved both at the configured parquet
  `balance_table_out` path and as
  `out_dir/tables/matching_balance_all_covariates.csv`; it is no longer limited
  to the top imbalance preview shown in the markdown summary.
