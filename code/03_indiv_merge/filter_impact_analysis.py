"""
filter_impact_analysis.py
=========================
Diagnoses how many candidate rows (and distinct FOIA applicants) each hard filter
in the indiv_merge pipeline drops.

Filters (in pipeline order):
  Stage 0 — raw cross-join (baseline)
  Stage 1 — match_order filters (applied inside stage_match_order):
      A. Gender       : f_score >= 0.20  (1 - F_PROB_BUFFER=0.8)
      B. YOB          : |yob - est_yob| <= 5, OR est_yob IS NULL and yob <= max_yob
      C. Date start   : YEAR(first_startdate) - (lottery_year-1) BETWEEN -4 AND 0
      D. Date end     : YEAR(last_enddate) >= lottery_year - 1
      E. Grad age     : last_grad_year - yob BETWEEN 15 AND 35 (or null)
      F. Grad recency : grad_years_since <= 4 (or null)
      G. STEM/occ     : stem_ind IS NULL OR stem_ind=1 OR foia_occ_ind IS NULL OR foia_occ_ind=1
      H. Dedup        : keep only best row per (foia_indiv_id, user_id) — match_order_ind=1
  Stage 2 — match_filt country filter (three-branch OR):
      I. Country/no-country/uncertain: country_score > 0.03 OR fallback conditions
  Stage 3 — bad_match_guard (applied in stage_final):
      J. Bad-match guard: NOT (country_score=0 AND subregion<0.15 AND weak gender/yob AND total<0.10)

Run interactively in iPython after importing indiv_merge to reuse its DuckDB connection.
"""

import time
import sys
import os

# ── reuse the module-level connection and registered tables from indiv_merge ──
sys.path.insert(0, os.path.dirname(__file__))
import indiv_merge as im
import indiv_merge_config as icfg

con = im.con_indiv

# ── parameters (mirror build_reg_inputs defaults) ────────────────────────────
F_PROB_BUFFER          = 0.8        # → f_score >= 0.20
YOB_BUFFER             = 5
GRAD_YEAR_BUFFER       = (15, 35)
COUNTRY_SCORE_CUTOFF   = 0.03
COUNTRY_FALLBACK_TOPN  = 3
COUNTRY_TOTAL_MARGIN   = 0.08
NO_COUNTRY_MIN_SUBREGION   = icfg.BUILD_NO_COUNTRY_MIN_SUBREGION_SCORE   # 0.20
NO_COUNTRY_MIN_TOTAL       = icfg.BUILD_NO_COUNTRY_MIN_TOTAL_SCORE        # 0.12
NO_COUNTRY_MIN_F_IF_NO_YOB = icfg.BUILD_NO_COUNTRY_MIN_F_SCORE_IF_EST_YOB_NULL  # 0.30
BAD_GUARD_SUBREGION    = icfg.BUILD_BAD_MATCH_GUARD_SUBREGION_SCORE_LT   # 0.15
BAD_GUARD_F_SCORE      = icfg.BUILD_BAD_MATCH_GUARD_F_SCORE_LT           # 0.30
BAD_GUARD_TOTAL        = icfg.BUILD_BAD_MATCH_GUARD_TOTAL_SCORE_LT       # 0.10
COUNTRY_UNCERTAIN_EXPR = im.COUNTRY_UNCERTAIN_EXPR   # "COALESCE(country_uncertain_ind, 0)"
ALPHA                  = icfg.BUILD_SUBREGION_BOOST_ALPHA
COMPETITION_WEIGHT     = icfg.BUILD_COUNTRY_COMPETITION_WEIGHT
COMPETITION_THRESHOLD  = icfg.BUILD_COUNTRY_COMPETITION_THRESHOLD

def count(query, label):
    """Run a count query and print results."""
    t0 = time.time()
    df = con.sql(f"SELECT COUNT(*) AS n_rows, COUNT(DISTINCT foia_indiv_id) AS n_apps FROM ({query})").df()
    elapsed = time.time() - t0
    n_rows = int(df["n_rows"].iloc[0])
    n_apps = int(df["n_apps"].iloc[0])
    print(f"  {label:<55} rows={n_rows:>12,}   apps={n_apps:>9,}   ({elapsed:.1f}s)")
    return n_rows, n_apps


print("="*85)
print("FILTER IMPACT ANALYSIS — indiv_merge hard filters")
print("="*85)

# ── Stage 0: materialise the raw cross-join ───────────────────────────────────
print("\nMaterialising _merge_raw_base (this may take a few minutes)...")
t0 = time.time()
raw_query = im.merge_raw_func(
    "rev_indiv", "foia_indiv",
    foia_prefilt    = "",
    subregion       = True,
    ALPHA           = ALPHA,
    COMPETITION_WEIGHT    = COMPETITION_WEIGHT,
    COMPETITION_THRESHOLD = COMPETITION_THRESHOLD,
)
im.materialize_table("_filt_diag_raw", raw_query, con=con)
print(f"  Done in {time.time()-t0:.1f}s")

print("\n--- Row/app counts at each filter stage ---\n")

raw_q = "SELECT * FROM _filt_diag_raw"
n0_rows, n0_apps = count(raw_q, "0. Raw cross-join (no filters)")

# ── Stage 1A: gender ──────────────────────────────────────────────────────────
q_A = f"SELECT * FROM _filt_diag_raw WHERE f_score >= 1 - {F_PROB_BUFFER}"
n_A_rows, n_A_apps = count(q_A, f"A. + gender (f_score >= {1-F_PROB_BUFFER:.2f})")

# ── Stage 1B: YOB ─────────────────────────────────────────────────────────────
q_B = f"""
SELECT * FROM _filt_diag_raw
WHERE f_score >= 1 - {F_PROB_BUFFER}
  AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
"""
n_B_rows, n_B_apps = count(q_B, f"B. + YOB (within {YOB_BUFFER} yrs or null)")

# ── Stage 1C: career start date ───────────────────────────────────────────────
q_C = f"""
SELECT * FROM _filt_diag_raw
WHERE f_score >= 1 - {F_PROB_BUFFER}
  AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
  AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
"""
n_C_rows, n_C_apps = count(q_C, "C. + date start (first job within window)")

# ── Stage 1D: still employed ──────────────────────────────────────────────────
q_D = f"""
SELECT * FROM _filt_diag_raw
WHERE f_score >= 1 - {F_PROB_BUFFER}
  AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
  AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
  AND YEAR(last_enddate) >= (lottery_year::INT - 1)
"""
n_D_rows, n_D_apps = count(q_D, "D. + date end (still active at lottery year)")

# ── Stage 1E: graduation age ──────────────────────────────────────────────────
q_E = f"""
SELECT * FROM _filt_diag_raw
WHERE f_score >= 1 - {F_PROB_BUFFER}
  AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
  AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
  AND YEAR(last_enddate) >= (lottery_year::INT - 1)
  AND (last_grad_year IS NULL OR (last_grad_year::INTEGER - yob::INTEGER) BETWEEN {GRAD_YEAR_BUFFER[0]} AND {GRAD_YEAR_BUFFER[1]})
"""
n_E_rows, n_E_apps = count(q_E, f"E. + grad age (grad - yob in [{GRAD_YEAR_BUFFER[0]},{GRAD_YEAR_BUFFER[1]}] or null)")

# ── Stage 1F: graduation recency ──────────────────────────────────────────────
q_F = f"""
SELECT * FROM _filt_diag_raw
WHERE f_score >= 1 - {F_PROB_BUFFER}
  AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
  AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
  AND YEAR(last_enddate) >= (lottery_year::INT - 1)
  AND (last_grad_year IS NULL OR (last_grad_year::INTEGER - yob::INTEGER) BETWEEN {GRAD_YEAR_BUFFER[0]} AND {GRAD_YEAR_BUFFER[1]})
  AND (grad_years_since IS NULL OR grad_years_since <= 4)
"""
n_F_rows, n_F_apps = count(q_F, "F. + grad recency (<= 4 yrs since grad or null)")

# ── Stage 1G: STEM / occ filter ───────────────────────────────────────────────
q_G = f"""
SELECT * FROM _filt_diag_raw
WHERE f_score >= 1 - {F_PROB_BUFFER}
  AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
  AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
  AND YEAR(last_enddate) >= (lottery_year::INT - 1)
  AND (last_grad_year IS NULL OR (last_grad_year::INTEGER - yob::INTEGER) BETWEEN {GRAD_YEAR_BUFFER[0]} AND {GRAD_YEAR_BUFFER[1]})
  AND (grad_years_since IS NULL OR grad_years_since <= 4)
  AND (stem_ind IS NULL OR stem_ind = 1 OR foia_occ_ind IS NULL OR foia_occ_ind = 1)
"""
n_G_rows, n_G_apps = count(q_G, "G. + STEM/occ filter")

# ── Stage 1H: dedup to best (foia_indiv_id, user_id) row ─────────────────────
# match_order_ind=1 keeps only the highest-scoring position-country combo per pair
q_H = f"""
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER(
            PARTITION BY foia_indiv_id, user_id
            ORDER BY
                CASE WHEN {COUNTRY_UNCERTAIN_EXPR} = 1
                     THEN 0.6*country_score + 0.4*subregion_score
                     ELSE country_score END DESC,
                subregion ASC,
                rcid ASC
        ) AS match_order_ind
    FROM _filt_diag_raw
    WHERE f_score >= 1 - {F_PROB_BUFFER}
      AND (ABS(yob::INTEGER - est_yob) <= {YOB_BUFFER} OR (est_yob IS NULL AND yob::INTEGER <= max_yob))
      AND YEAR(first_startdate) - (lottery_year::INT - 1) BETWEEN -4 AND 0
      AND YEAR(last_enddate) >= (lottery_year::INT - 1)
      AND (last_grad_year IS NULL OR (last_grad_year::INTEGER - yob::INTEGER) BETWEEN {GRAD_YEAR_BUFFER[0]} AND {GRAD_YEAR_BUFFER[1]})
      AND (grad_years_since IS NULL OR grad_years_since <= 4)
      AND (stem_ind IS NULL OR stem_ind = 1 OR foia_occ_ind IS NULL OR foia_occ_ind = 1)
)
SELECT * FROM ranked WHERE match_order_ind = 1
"""
im.materialize_table("_filt_diag_after_match_order", q_H, con=con)
n_H_rows, n_H_apps = count("SELECT * FROM _filt_diag_after_match_order", "H. + dedup (best row per app×user pair)")

# ── Stage 2I: country / no-country / uncertain filter ────────────────────────
q_I = f"""
WITH base AS (
    SELECT *,
        MAX(country_score) OVER(PARTITION BY foia_indiv_id) AS max_country_score_app,
        MAX(total_score)   OVER(PARTITION BY foia_indiv_id) AS max_total_score_app,
        ROW_NUMBER() OVER(
            PARTITION BY foia_indiv_id
            ORDER BY
                CASE WHEN {COUNTRY_UNCERTAIN_EXPR} = 1
                     THEN 0.6*country_score + 0.4*subregion_score
                     ELSE country_score END DESC,
                subregion ASC, rcid ASC, user_id ASC
        ) AS app_country_rank
    FROM _filt_diag_after_match_order
)
SELECT * FROM base
WHERE country_score > {COUNTRY_SCORE_CUTOFF}
   OR (
        max_country_score_app <= {COUNTRY_SCORE_CUTOFF}
        AND subregion_score >= {NO_COUNTRY_MIN_SUBREGION}
        AND total_score >= {NO_COUNTRY_MIN_TOTAL}
        AND (est_yob IS NOT NULL OR f_score >= {NO_COUNTRY_MIN_F_IF_NO_YOB})
   )
   OR (
        {COUNTRY_UNCERTAIN_EXPR} = 1
        AND (
            app_country_rank <= {COUNTRY_FALLBACK_TOPN}
            OR total_score >= max_total_score_app - {COUNTRY_TOTAL_MARGIN}
        )
   )
"""
im.materialize_table("_filt_diag_after_country_filt", q_I, con=con)
n_I_rows, n_I_apps = count(
    "SELECT * FROM _filt_diag_after_country_filt",
    f"I. + country filter (country_score > {COUNTRY_SCORE_CUTOFF} or fallback)"
)

# ── Stage 3J: bad-match guard ─────────────────────────────────────────────────
q_J = f"""
SELECT * FROM _filt_diag_after_country_filt
WHERE NOT (
    country_score = 0
    AND subregion_score < {BAD_GUARD_SUBREGION}
    AND (est_yob IS NULL OR f_score < {BAD_GUARD_F_SCORE})
    AND total_score < {BAD_GUARD_TOTAL}
)
"""
n_J_rows, n_J_apps = count(q_J, "J. + bad-match guard")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*85)
print("SUMMARY: incremental rows dropped by each filter")
print("="*85)
print(f"{'Step':<55} {'Rows dropped':>14}  {'%':>6}  {'Apps lost':>10}  {'%':>6}")
print("-"*85)

stages = [
    ("0. Raw cross-join",                                         n0_rows,  n0_apps),
    ("A. Gender (f_score >= 0.20)",                               n_A_rows, n_A_apps),
    ("B. YOB (within 5 yrs or null)",                             n_B_rows, n_B_apps),
    ("C. Date start (first job in window)",                        n_C_rows, n_C_apps),
    ("D. Date end (still active)",                                 n_D_rows, n_D_apps),
    ("E. Grad age (grad-yob in [15,35] or null)",                 n_E_rows, n_E_apps),
    ("F. Grad recency (<= 4 yrs or null)",                        n_F_rows, n_F_apps),
    ("G. STEM/occ filter",                                         n_G_rows, n_G_apps),
    ("H. Dedup (best row per app×user)",                           n_H_rows, n_H_apps),
    ("I. Country filter (score>0.03 or fallback)",                 n_I_rows, n_I_apps),
    ("J. Bad-match guard",                                         n_J_rows, n_J_apps),
]

for i in range(1, len(stages)):
    label    = stages[i][0]
    prev_r, prev_a = stages[i-1][1], stages[i-1][2]
    curr_r, curr_a = stages[i][1],   stages[i][2]
    dr = prev_r - curr_r
    da = prev_a - curr_a
    pct_r = 100 * dr / prev_r if prev_r > 0 else 0
    pct_a = 100 * da / prev_a if prev_a > 0 else 0
    print(f"  {label:<53} {dr:>14,}  {pct_r:>5.1f}%  {da:>10,}  {pct_a:>5.1f}%")

print("-"*85)
total_dr = n0_rows - n_J_rows
total_da = n0_apps - n_J_apps
print(f"  {'TOTAL reduction':53} {total_dr:>14,}  {100*total_dr/n0_rows:>5.1f}%  {total_da:>10,}  {100*total_da/n0_apps:>5.1f}%")
print(f"\nFinal: {n_J_rows:,} rows for {n_J_apps:,} apps  "
      f"(avg multiplicity: {n_J_rows/n_J_apps:.1f}x)")

# ── cleanup ───────────────────────────────────────────────────────────────────
for t in ["_filt_diag_raw", "_filt_diag_after_match_order", "_filt_diag_after_country_filt"]:
    try:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    except Exception:
        pass
print("\nDone. Temporary diagnostic tables dropped.")
