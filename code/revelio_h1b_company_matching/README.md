
# Revelio ↔ FOIA Employer Dedup + Crosswalk (Python)

This folder contains a small, reproducible pipeline for:

1) **Normalizing / cleaning company names** in both datasets  
2) **Deduplicating companies within each dataset** (conservative clustering)  
3) **Building a FOIA↔Revelio crosswalk** and assigning a **universal firm identifier**  

It is inspired by the token-based blocking + string-distance logic in your legacy `employer_merge.R`, and supports optionally re-using the detailed “alias rules” in `firm_names_preclean.do`.

## Files

### Core modules
- `company_name_cleaning.py`  
  General name normalization + NAICS/state normalization + optional Stata `.do` alias rules parser.
- `dedupe_utils.py`  
  Blocking + duplicate-edge creation + union-find components + canonical aggregation.

### CLI scripts
- `dedupe_foia_employers.py`  
  Builds a **FEIN-level employer table** from FOIA data, then deduplicates across FEINs into “FOIA firms”.
- `dedupe_revelio_companies.py`  
  Builds an **RCID-level company table**, then deduplicates across RCIDs into “Revelio firms”.
- `build_foia_revelio_crosswalk.py`  
  Builds **candidate matches**, extracts **mutual-best matches**, then assigns **universal firm IDs** via connected components.

## Installation / requirements

These scripts are pure Python and rely on common packages:
- pandas
- rapidfuzz
- unidecode
- networkx (optional, but recommended)

Example:
```bash
pip install pandas rapidfuzz unidecode networkx
```

## Typical workflow

### 1) Deduplicate FOIA employers

```bash
python dedupe_foia_employers.py \
  --foia /path/to/foia.csv \
  --outdir /path/to/out/foia \
  --fein-col FEIN \
  --name-col employer_name \
  --hq-state-col state \
  --work-state-col WORKSITE_STATE \
  --naics-col NAICS_CODE \
  --multireg-col ben_multi_reg_ind \
  --selected-col status_type \
  --selected-value SELECTED \
  --stata-rules /path/to/firm_names_preclean.do
```

Outputs:
- `foia_fein_entities.csv` (1 row/FEIN)
- `foia_firms_dedup.csv` (1 row/deduped FOIA firm)
- `foia_fein_to_firm.csv` (FEIN → FOIA firm)
- `foia_dedupe_edges.csv` (debug duplicate edges)

### 2) Deduplicate Revelio companies

```bash
python dedupe_revelio_companies.py \
  --revelio /path/to/companies_by_positions_locations.csv \
  --outdir /path/to/out/revelio \
  --rcid-col rcid \
  --name-col company \
  --top-state-col top_state \
  --naics-col naics_code \
  --lei-col lei \
  --weight-col n_users \
  --use-lei \
  --stata-rules /path/to/firm_names_preclean.do
```

Outputs:
- `revelio_rcid_entities.csv`
- `revelio_firms_dedup.csv`
- `revelio_rcid_to_firm.csv`
- `revelio_dedupe_edges.csv`

### 3) Build crosswalk + universal IDs

```bash
python build_foia_revelio_crosswalk.py \
  --foia-firms /path/to/out/foia/foia_firms_dedup.csv \
  --revelio-firms /path/to/out/revelio/revelio_firms_dedup.csv \
  --foia-fein-to-firm /path/to/out/foia/foia_fein_to_firm.csv \
  --revelio-rcid-to-firm /path/to/out/revelio/revelio_rcid_to_firm.csv \
  --outdir /path/to/out/crosswalk
```

Optional: if you have **Revelio employee work locations** as a long file (e.g., `rcid, state, n`), you can pass:

```bash
  --revelio-locations /path/to/revelio_locations.csv \
  --revelio-loc-rcid-col rcid \
  --revelio-loc-state-col state \
  --revelio-loc-weight-col n
```

Outputs:
- `crosswalk_candidates.csv` (top candidates per FOIA firm)
- `crosswalk_mutual_best.csv` (mutual-best matches above threshold)
- `universal_companies.csv` (universal firm entities)
- `foia_firm_to_universal.csv`
- `revelio_firm_to_universal.csv`
- `foia_fein_to_universal.csv` (if FEIN mapping provided)
- `revelio_rcid_to_universal.csv` (if RCID mapping provided)

### 4) Compile LLM review outputs (optional)

After running the LLM review step (`run_crosswalk_llm_review.py`), compile JSON outputs into a cleaned match table:

```bash
python compile_llm_matches.py \
  --config /path/to/company_review_bf.yaml \
  --candidates /path/to/out/crosswalk/crosswalk_candidates.csv \
  --llm-output-dir /path/to/llm_outputs \
  --min-confidence 0.5 \
  --require-valid
```

Outputs:
- `llm_review_all.csv` (all LLM judgments)
- `llm_review_matches.csv` (filtered matches)

## Tuning

The pipeline is **conservative by default**.

- Within-source dedup thresholds are in:
  - `dedupe_foia_employers.py` (CLI args `--dedupe-*`)
  - `dedupe_revelio_companies.py` (CLI args `--dedupe-*`)
- Crosswalk thresholds are in:
  - `build_foia_revelio_crosswalk.py` (CLI args `--candidate-threshold` and `--high-threshold`)

If you want “more aggressive” merging, lower thresholds slightly, but beware of collapsing subsidiaries.

## Notes / differences vs legacy R

- The candidate generation uses **rare-token blocking**, similar in spirit to your R workflow (compute token frequencies, drop common tokens, use rare tokens to generate candidate matches).
- The new code explicitly produces:
  - **within-source firm IDs** (FOIA firm UID; Revelio firm UID)
  - a **universal UID** (one-to-one for mutual best matches; but it’s built as connected components so it can gracefully handle occasional many-to-one clusters)
- All IDs are **deterministic hash IDs** built from the underlying members of each cluster.
