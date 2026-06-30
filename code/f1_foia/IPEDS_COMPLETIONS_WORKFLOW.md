# IPEDS Completions Workflow

Build the raw IPEDS completions panel from NCES source files:

```bash
python code/f1_foia/ipeds_completions_download.py
```

By default this downloads or reuses:

- `C{year}_A.zip` / `C{year}_A_Data_Stata.zip` for 2000-2024
- `C{year}DEP.zip` / `C{year}DEP_Data_Stata.zip` for 2013-2024
- `Crosswalk2000to2010.csv`
- `Crosswalk2010to2020.csv`

Default outputs:

- `{root}/data/raw/ipeds_completions_all.dta`
- `{root}/data/raw/ipeds/completions_all.dta`
- `{root}/data/raw/ipeds/ipeds_completions_panel.parquet`
- `{root}/data/raw/ipeds/ipeds_completions_labels.json`

Then build the cleaned analysis parquet:

```bash
python code/f1_foia/ipeds_clean.py
```

which writes:

```text
{root}/data/int/int_files_nov2025/ipeds_completions_all.parquet
```

Useful smoke-test command:

```bash
python code/f1_foia/ipeds_completions_download.py \
  --start-year 2024 \
  --dep-start-year 2024 \
  --end-year 2024 \
  --skip-dta \
  --skip-historical-dta \
  --panel-parquet /private/tmp/ipeds_completions_2024_test.parquet \
  --labels-json /private/tmp/ipeds_completions_2024_labels.json
```
