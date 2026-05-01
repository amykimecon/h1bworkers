# High-Level Identification Strategies Tried in `company_shift_share`

## Scope
This note summarizes the empirical designs used across the main analysis scripts in `company_shift_share`. It focuses on broad identification ideas and their main variants, and leaves out implementation details, data engineering, and code-level diagnostics.

At a high level, the project moves through three broad stages:

1. school-shock shift-share designs,
2. event studies that sharpen timing and comparison groups,
3. a newer direct firm-exposure design built from source data.

## 1. Shift-Share Designs Based on School Shocks
**Core idea:** A firm is more exposed if, before the policy period, it hired more heavily from schools that later experienced large changes in OPT-related or international-student flows.

### Main variants
- **Broad-school design:** Build exposure using the full school universe.
- **Matched-school design:** Restrict the school side to treated/control school pairs chosen to look similar on observable characteristics, then build firm exposure from only those schools.
- **Persistent step shock:** Treat a school shock as something that stays in place after it arrives.
- **One-year pulse shock:** Treat the shock as concentrated in the event year only.
- **Cumulative post-shock design:** Let exposure continue to evolve with the school’s post-event path rather than only the initial break.
- **Predicted-OPT-weighted design:** Scale school shocks by how OPT-oriented a school looked before the event.
- **Composition-only design:** Focus on changes in composition rather than changes scaled by school size.
- **Alternative share windows:** Compute firm-school exposure using an earlier pre-period, a later pre-period, a longer pre-period, or the full sample.
- **Placebo timing:** Move the same school shocks earlier in time to check for spurious pre-trends.

### Other important variations inside this family
- School shocks were built from either OPT-related measures or broader international-student measures.
- Some versions changed the denominator used to normalize school shocks, making the exposure measure more about counts versus rates.
- Degree scope also varied, especially between master’s-only and broader bachelor’s-plus-master’s definitions.

This family is the main legacy instrument-based approach and feeds the first-stage, reduced-form, and instrumental-variables exercises.

## 2. Common-Year Event Studies Around the 2015-2016 OPT Policy Change
**Core idea:** Treat the policy change as a single calendar shock and compare firms that looked more versus less exposed beforehand.

### Main variants
- **Raw policy-timing event study:** Align every firm to the same calendar event year and trace outcomes before and after that date.
- **Within-firm event study:** Compare each firm to itself relative to a pre-policy reference year.
- **Exposure-group design:** Sort firms into low- versus high-exposure groups and compare their paths around the policy break.
- **Continuous-exposure design:** Use exposure as a continuous intensity rather than coarse groups.

### Exposure measures used here
- Pre-period OPT hiring rates.
- Pre-period shares of hires from OPT-intensive schools.
- A modeled firm-level exposure index built from pre-period firm characteristics.

### Outcomes emphasized
- OPT hiring itself.
- Overall employment and new hiring.
- Native/foreign splits of those outcomes in the newer source-based pipeline.

## 3. Firm-Specific Shock-Timing Event Studies
**Core idea:** Instead of forcing all firms to share the same event date, assign treatment timing based on when a firm experiences a sharp rise in its own exposure measure.

### Main variants
- **Absorbing-event design:** Give each firm one main event, usually its largest isolated positive shock.
- **Event-level design:** Let a firm contribute more than one treated event if the events are far enough apart.
- **Percent-change timing:** Define shocks using sharp percentage jumps in exposure.
- **Level-change timing:** Define shocks using sharp absolute jumps instead.
- **Noise-screened timing:** Require the event to be locally isolated so small nearby movements do not count as separate treatment events.

### Estimation variants
- Standard two-way fixed-effects event studies.
- Newer staggered-timing estimators designed to be safer when treatment effects vary across cohorts and over time.
- Dynamic event-time profiles as well as simpler post-treatment summaries.

This is the main attempt to make timing endogenous to the firm’s own shock path rather than to an economy-wide policy date.

## 4. Matched Comparison Designs
**Core idea:** Improve comparability by explicitly matching exposed firms to similar firms that do not experience a nearby shock.

### Main variants
- **Event-time matched controls:** For each treated firm-year, choose controls from the same calendar year whose exposure path stays quiet nearby.
- **Match on recent outcomes:** Use lagged firm size and recent OPT hiring so treated and control firms start from similar levels.
- **Dominant-school matching:** Collapse each firm to its main school pipeline and compare firms tied to different schools with different shock timing.
- **Shock-timing pair design:** Form pairs of firms with similar initial size but sufficiently different timing of their main school shock.

### What these designs are trying to solve
- Reduce reliance on broad cross-firm comparisons.
- Make treatment timing more transparent.
- Put more weight on “like-with-like” comparisons rather than only fixed effects.

## 5. Direct Firm-Exposure Models Built from Source Data
**Core idea:** Replace the original school-shock instrument with a directly estimated firm-level propensity to use OPT, using only pre-2016 firm characteristics, and then carry that exposure into an event study around the policy change.

### Main variants
- **Legacy exposure measures kept in parallel:** Prior OPT hiring rates and prior hiring from OPT-intensive schools remain as benchmarks.
- **Modeled exposure index:** Predict which firms are most likely to use OPT after 2016 using pre-period firm characteristics.
- **Grouped versus continuous exposure:** Enter the predicted exposure either as bins or as a continuous index.
- **Binary versus count targets:** Predict either whether a firm uses OPT at all after 2016 or how much it uses OPT.
- **Alternative model classes:** Compare simple linear models, logistic models, penalized models, count models, and random forests.
- **Holdout versus full-sample evaluation:** Check whether the index works only in-sample or also on firms left out of training.
- **Expanded comparison sample:** Add clearly low-exposure firms from outside the core source sample to sharpen separation in the prediction exercise.

This is the clearest break from the older design: it no longer relies on the original shift-share pipeline as the upstream source of treatment intensity.

## 6. Supporting Measurement and Validation Work
Some scripts support the designs above without introducing a separate identification strategy:

- **Company-linkage comparisons:** Compare old and new employer matching so the firm definition is more credible.
- **School and degree audits:** Check how schools and degree groups are classified.
- **Model-comparison exercises:** Compare alternative prediction targets and model families for the exposure index.
- **Instrument decomposition and balance checks:** Examine which schools drive the instrument and whether exposure is already correlated with pre-period firm characteristics.

## Bottom Line
The progression in this folder is fairly clear. It starts with a school-based shift-share design, then tries to sharpen that design through matched samples and firm-specific event timing, and finally moves to a direct firm-level exposure approach built from source data. The main recurring questions are:

- how to define exposure,
- how to define the event date,
- and how to choose the comparison group.

Most of the script-level variation is a different answer to one of those three questions.
