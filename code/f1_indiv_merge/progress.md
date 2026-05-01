# f1_indiv_merge Pipeline — Progress

**Config:** `f1_indiv_merge/pipeline.yaml`

> **Status key:** `☐` not started · `~` in progress · `✓` complete · `↩` needs rerun
>
> To mark a stage complete, run the full non-testing stage entrypoint or `run_all.py`
> — the tracker updates this file automatically and records wall-clock runtime.
> Reset a row manually (change `✓` → `↩`, clear the date/by/duration) whenever a
> stage changes and you need to rerun it.

---

## Stage 01 · F1 FOIA Clean

> **Rerun if:** raw FOIA input changes, stage-01 code changes, or shared config/runtime
> behavior changes in a way that affects the raw combine or person-linkage outputs.

| Stage              | Status | Completed at     | By     | Duration  | Notes                                                                    |
| -------------------|--------|------------------|--------|-----------|------------------------------------------------------------------------- |
| `01_f1_foia_clean` | ✓      | 2026-04-02 21:48 | yk0581 | 4h 0m 33s | Builds combined raw FOIA, person-id crosswalk, cleaned FOIA person panel |

---

## Stage 02 · Revelio Import

> **Rerun if:** cleaned FOIA school inputs change, stage-02 query/filter logic changes,
> shard settings change, or WRDS pulls need to be regenerated.

| Stage           | Status | Completed at     | By     | Duration  | Notes                                                                                    |
| ----------------|--------|------------------|--------|-----------|----------------------------------------------------------------------------------------- |
| `02_rev_import` | ✓      | 2026-04-03 10:14 | yk0581 | 11h 4m 7s | Builds school regex artifacts, range shard manifest, matched-user outputs, WRDS user/pos |

---

## Stage 03 · Revelio Crosswalks

> **Rerun if:** upstream cleaned education artifacts change, stage-03 crosswalk logic
> changes, or supporting reference/crosswalk inputs change.

| Stage               | Status | Completed at     | By     | Duration | Notes                                                                                                 |
| --------------------|--------|------------------|--------|----------|------------------------------------------------------------------------------------------------------ |
| `03_rev_crosswalks` | ✓      | 2026-04-15 22:56 | yk0581 | 2s       | Implemented locally; school/employer smoke-tested, field crosswalk still needs a full real-data rerun |

---

## Stage 04 · Revelio User Clean

> **Rerun if:** stage-02 or stage-03 outputs change, or stage-04 cleaning/model logic changes.

| Stage               | Status | Completed at     | By     | Duration | Notes                                                                                                           |
| --------------------|--------|------------------|--------|----------|---------------------------------------------------------------------------------------------------------------- |
| `04_rev_user_clean` | ✓      | 2026-04-16 19:35 | yk0581 | 6m 19s   | Implemented locally; builds name-model outputs, cleaned user/education/position artifacts, and match-ready rows |

---

## Stage 05 · Final Individual Merge

> **Rerun if:** any upstream stage output changes, merge scoring logic changes, or the
> current legacy-wrapper behavior changes.

| Stage            | Status | Completed at     | By     | Duration   | Notes                                                                                                        |
| -----------------|--------|------------------|--------|------------|------------------------------------------------------------------------------------------------------------- |
| `05_indiv_merge` | ✓      | 2026-04-18 02:37 | yk0581 | 3h 52m 57s | Implemented locally; now reads pipeline.yaml + stage-03/04 artifacts, but still needs a full real-data rerun |
