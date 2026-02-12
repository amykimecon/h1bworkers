# Company-Level Shift-Share Pipeline

This folder holds code to build company-level (RCID) shift-share instruments that mirror the IHMA logic but swap the unit of observation from origin institutions to employers. Data stay in `root/data`—this repo only stores code.

## Overview
- **Instrument**: `z_ct = Σ_k s_ck * g_kt`, where `s_ck` is the base-year share of new hires at company `c` from university `k` (built from `revelio_school_to_employer_transitions.parquet` mapped to UNITIDs) and `g_kt` is recomputed from IPEDS using the IHMA procedure.
- **Treatment**: Master’s OPT hires by company and hire year (F-1 FOIA + employer crosswalk).
- **Outcome**: U.S. employee counts by company and outcome year from Revelio headcounts; mapped to hire year via a lag range (multiple lags land in the analysis panel with suffixes).

## Inputs (expected under `root/data`)
- `int/int_files_jan2026/revelio_school_to_employer_transitions.parquet`
- `int/int_files_jan2026/revelio_company_headcounts.parquet`
- `int/rsid_ipeds_cw.parquet` (RSID → UNITID)
- `int/ipeds_ma_only.parquet` (for `g_kt`)
- `int/foia_sevp_combined_raw.parquet`
- `int/int_files_jan2026/f1_employer_final_crosswalk.parquet`

## Outputs (written to `root/data/out/company_shift_share`)
- `instrument_components.parquet`: company×university×t components with `s_ckr`, `g_kt`, and products.
- `instrument_panel.parquet`: `z_ct` aggregated to company×t.
- `masters_opt_hires.parquet`: treatment counts by company×t.
- `outcomes.parquet`: long outcomes with outcome year `s`, hire year `t`, employee counts, and lag.
- `analysis_panel.parquet`: merged panel with `z_ct`, treatment, and lagged outcomes (e.g., `y_cst_lag0` ... `y_cst_lag5`).

## Config
All scripts accept `--config` (default: `configs/company_shift_share.yaml`). The YAML centralizes paths and parameters so the same config can be reused across all steps.

## Dependency builders
```
python company_shift_share/build_all_deps.py
```

You can skip steps:
```
python company_shift_share/build_all_deps.py --skip-revelio
```

Individual dependency scripts (all accept `--config`):
- `company_shift_share/deps_foia_clean.py`
- `company_shift_share/deps_foia_person_id_linkage.py`
- `company_shift_share/deps_rsid_ipeds_cw.py`
- `company_shift_share/deps_ipeds_ma_only.py`

## Running
```
python -m company_shift_share.build_company_shift_share \
  --outcome-lag-start 0 \
  --outcome-lag-end 5 \
  --share-base-year 2010

python -m company_shift_share.build_company_shift_share \
  --outcome-lag 2   # single lag override (same as start=end=2)
```

### Notes
- `g_kt` is recomputed from IPEDS using the same rules as `ihma_clean.py` (international-heavy master’s programs, filled annual series, `t = year - 2`).
- Master’s-only filtering in FOIA is applied when relevant columns exist (`awlevel_group`, `awlevel`, `education_level`, or `degree_level`); otherwise the script falls back to all OPT entries with a warning.
