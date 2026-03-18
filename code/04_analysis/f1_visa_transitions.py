"""
f1_visa_transitions.py
----------------------
Computes, by F-1 entry cohort year, the share of F-1 students who eventually
transition to each requested visa type, and of those, what share used OPT at
any point during their F-1 spell.

Unit of analysis: person_id (unique individual linked across SEVIS years)
Cohort definition: year of first appearance in the SEVIS data (entry cohort)

Outputs:
  - output/f1_transitions/f1_transition_by_cohort.csv  (wide: one row per cohort-year)
  - output/f1_transitions/f1_transition_long.csv        (long: one row per cohort × visa type)
  - output/f1_transitions/fig_transition_shares.png
  - output/f1_transitions/fig_opt_by_visa.png
  - Printed tables to stdout

Notes:
  - requested_status is a static field per person (their eventual visa transition)
  - OPT flag: person ever has employment_opt_type IS NOT NULL
  - Recent cohorts (2021+) are right-censored and will show lower transition rates
  - Persons with NULL person_id are excluded (not linked across years)
"""

import os
import time
import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Paths ────────────────────────────────────────────────────────────────────
F1_PARQUET = (
    "/home/yk0581/data/int/int_files_feb2026/"
    "foia_sevp_with_person_id_employment_corrected.parquet"
)
OUT_DIR = "/home/yk0581/h1bworkers/output/f1_transitions"
os.makedirs(OUT_DIR, exist_ok=True)

# Visa types to report individually (others grouped as "Other")
VISA_TYPES_KEEP = ["H1B", "1B1", "O1B", "O1A", "H4", "R1", "E2", "E3", "TN1", "TN2", "J1"]

# ── Connect ───────────────────────────────────────────────────────────────────
t0 = time.time()
con = duckdb.connect()
print("Building person-level summary from F-1 SEVIS data...")

# ── Step 1: Person-level summary ──────────────────────────────────────────────
# For each person_id, extract:
#   - entry cohort year (MIN year_int)
#   - whether they ever used OPT (any non-null employment_opt_type)
#   - what visa type they transitioned to (requested_status, MODE if varies)
#   - OPT types used (for sub-analysis)
person_df = con.execute("""
    SELECT
        person_id,
        MIN(year_int)                                           AS cohort_year,
        -- OPT: any record with non-null employment_opt_type
        MAX(CASE WHEN employment_opt_type IS NOT NULL THEN 1 ELSE 0 END) AS had_opt,
        -- STEM OPT specifically
        MAX(CASE WHEN employment_opt_type = 'STEM' THEN 1 ELSE 0 END)  AS had_stem_opt,
        -- Transition visa (static for most persons; take first non-null if any)
        MIN(requested_status)                                   AS requested_status,
        -- Final F-1 status
        MAX(CASE WHEN status_code = 'COMPLETED'   THEN 1 ELSE 0 END) AS ever_completed,
        MAX(CASE WHEN status_code = 'TERMINATED'  THEN 1 ELSE 0 END) AS ever_terminated,
        MAX(CASE WHEN status_code = 'DEACTIVATED' THEN 1 ELSE 0 END) AS ever_deactivated,
        -- Program dates
        MIN(program_start_date)                                 AS program_start_date,
        MAX(program_end_date)                                   AS program_end_date,
        -- Year range
        MIN(year_int)                                           AS year_first,
        MAX(year_int)                                           AS year_last,
        COUNT(DISTINCT year_int)                                AS n_years_observed
    FROM read_parquet(?)
    WHERE person_id IS NOT NULL
    GROUP BY person_id
""", [F1_PARQUET]).fetchdf()

n_persons = len(person_df)
print(f"  Total unique persons (with person_id): {n_persons:,}")
print(f"  Persons with any requested_status:     {person_df['requested_status'].notna().sum():,} "
      f"({person_df['requested_status'].notna().mean():.1%})")
print(f"  Persons who used OPT:                  {person_df['had_opt'].sum():,} "
      f"({person_df['had_opt'].mean():.1%})")
print(f"  Elapsed: {time.time()-t0:.1f}s")

# ── Step 2: Normalize requested_status ────────────────────────────────────────
# Bucket small visa types into "Other"
person_df["visa_type"] = person_df["requested_status"].apply(
    lambda x: x if (pd.notna(x) and x in VISA_TYPES_KEEP) else (
        "Other" if pd.notna(x) else None
    )
)

print("\nVisa transition counts across all cohorts:")
vc = person_df["visa_type"].value_counts(dropna=False)
print(vc.to_string())

# ── Step 3: By-cohort aggregation ─────────────────────────────────────────────
print("\nComputing shares by cohort year...")

# Restrict to cohorts 2006-2022 (2023 is severely right-censored in the data)
COHORT_MAX = 2022
df = person_df[person_df["cohort_year"] <= COHORT_MAX].copy()

# Cohort totals
cohort_totals = df.groupby("cohort_year").size().rename("n_persons_cohort")

# --- A. Share transitioning to any visa, by cohort ---
any_transition = (
    df.assign(transitioned=lambda d: d["visa_type"].notna().astype(int))
    .groupby("cohort_year")["transitioned"]
    .mean()
    .rename("share_any_transition")
)

# --- B. OPT usage among all F-1 students, by cohort ---
opt_share_all = (
    df.groupby("cohort_year")["had_opt"]
    .mean()
    .rename("share_opt_any")
)
stem_opt_share_all = (
    df.groupby("cohort_year")["had_stem_opt"]
    .mean()
    .rename("share_stem_opt_any")
)

# --- C. Share transitioning to each visa type, by cohort ---
visa_types_all = ["H1B", "1B1", "O1B", "O1A", "H4", "R1", "E2", "E3",
                  "TN1", "TN2", "J1", "Other"]

def share_visa(df, vtype, cohort_col="cohort_year"):
    return (
        df.assign(flag=lambda d: (d["visa_type"] == vtype).astype(int))
        .groupby(cohort_col)["flag"]
        .mean()
        .rename(f"share_{vtype.lower()}")
    )

visa_shares = pd.concat(
    [share_visa(df, v) for v in visa_types_all], axis=1
)

# --- D. OPT usage conditional on each visa transition, by cohort ---
def opt_share_given_visa(df, vtype):
    sub = df[df["visa_type"] == vtype]
    if len(sub) == 0:
        return pd.Series(dtype=float, name=f"opt_given_{vtype.lower()}")
    return (
        sub.groupby("cohort_year")["had_opt"]
        .mean()
        .rename(f"opt_given_{vtype.lower()}")
    )

opt_given_visa = pd.concat(
    [opt_share_given_visa(df, v) for v in ["H1B", "1B1", "O1B", "O1A", "H4"]],
    axis=1
)

# --- Assemble wide table ---
wide = pd.concat(
    [cohort_totals, any_transition, opt_share_all, stem_opt_share_all,
     visa_shares, opt_given_visa],
    axis=1
).reset_index()

print(f"\n{'='*70}")
print("WIDE TABLE: Transition shares by cohort year")
print(f"{'='*70}")
display_cols = ["cohort_year", "n_persons_cohort", "share_any_transition",
                "share_opt_any", "share_stem_opt_any",
                "share_h1b", "share_1b1", "share_o1b", "share_o1a",
                "share_h4", "share_tn1", "share_r1", "share_other"]
print(wide[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# --- Long table (for charting) ---
# One row per cohort × visa type
long_rows = []
for _, row in wide.iterrows():
    yr = row["cohort_year"]
    n = row["n_persons_cohort"]
    for v in visa_types_all:
        col = f"share_{v.lower()}"
        if col in wide.columns:
            long_rows.append({
                "cohort_year": yr,
                "n_persons_cohort": n,
                "visa_type": v,
                "share": row[col],
                "opt_given_visa": row.get(f"opt_given_{v.lower()}", float("nan"))
            })

long_df = pd.DataFrame(long_rows)

# ── Step 4: OPT share conditional on visa type (pooled) ───────────────────────
print(f"\n{'='*70}")
print("OPT USAGE BY VISA TRANSITION (pooled, cohorts 2006-2022)")
print(f"{'='*70}")

opt_by_visa = (
    df.groupby("visa_type", dropna=False)
    .agg(
        n_persons=("had_opt", "count"),
        share_had_opt=("had_opt", "mean"),
        share_had_stem_opt=("had_stem_opt", "mean"),
    )
    .sort_values("n_persons", ascending=False)
)
print(opt_by_visa.to_string(float_format=lambda x: f"{x:.3f}"))

# ── Step 5: Save CSVs ─────────────────────────────────────────────────────────
wide.to_csv(f"{OUT_DIR}/f1_transition_by_cohort.csv", index=False)
long_df.to_csv(f"{OUT_DIR}/f1_transition_long.csv", index=False)
opt_by_visa.to_csv(f"{OUT_DIR}/f1_opt_by_visa_pooled.csv")
print(f"\nSaved CSVs to {OUT_DIR}/")

# ── Step 6: Figures ───────────────────────────────────────────────────────────
PLOT_YEARS = list(range(2006, 2022))  # Exclude 2022 right-censoring endpoint
PLOT_VISAS = ["H1B", "1B1", "O1B", "O1A", "H4", "Other"]
COLORS = sns.color_palette("tab10", len(PLOT_VISAS))

# --- Fig 1: Share transitioning to each visa type by cohort year ---
fig, ax = plt.subplots(figsize=(11, 6))
plot_wide = wide[wide["cohort_year"].isin(PLOT_YEARS)]

for i, v in enumerate(PLOT_VISAS):
    col = f"share_{v.lower()}"
    if col in wide.columns:
        ax.plot(plot_wide["cohort_year"], plot_wide[col] * 100,
                marker="o", markersize=4, label=v, color=COLORS[i])

ax.set_xlabel("F-1 Entry Cohort Year")
ax.set_ylabel("Share of F-1 Students (%)")
ax.set_title("F-1 Visa Transition Rates by Entry Cohort")
ax.legend(title="Requested Visa", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax.grid(axis="y", alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_transition_shares.png", dpi=150)
plt.show()
print("Saved fig_transition_shares.png")

# --- Fig 2: OPT usage rate and STEM OPT by cohort year ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(plot_wide["cohort_year"], plot_wide["share_opt_any"] * 100,
        marker="o", markersize=4, label="Any OPT", color="steelblue")
ax.plot(plot_wide["cohort_year"], plot_wide["share_stem_opt_any"] * 100,
        marker="s", markersize=4, label="STEM OPT", color="darkorange")
ax.plot(plot_wide["cohort_year"], plot_wide["share_any_transition"] * 100,
        marker="^", markersize=4, label="Any Visa Transition", color="forestgreen",
        linestyle="--")
ax.set_xlabel("F-1 Entry Cohort Year")
ax.set_ylabel("Share of F-1 Students (%)")
ax.set_title("OPT Usage and Visa Transition Rates by F-1 Entry Cohort")
ax.legend()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
ax.grid(axis="y", alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_opt_rates.png", dpi=150)
plt.show()
print("Saved fig_opt_rates.png")

# --- Fig 3: OPT share among those who transitioned, by visa type (pooled bar) ---
opt_plot = opt_by_visa[opt_by_visa.index.notna()].copy().reset_index()
opt_plot = opt_plot.sort_values("share_had_opt", ascending=False).head(12)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(opt_plot))
ax.bar(x, opt_plot["share_had_opt"] * 100, color="steelblue",
       label="Any OPT", alpha=0.8)
ax.bar(x, opt_plot["share_had_stem_opt"] * 100, color="darkorange",
       label="STEM OPT", alpha=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(opt_plot["visa_type"], rotation=30)
ax.set_ylabel("Share who used OPT (%)")
ax.set_title("OPT Usage Rate Among F-1 Students, by Eventual Visa Transition (Pooled)")
ax.legend()
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax.grid(axis="y", alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_opt_by_visa.png", dpi=150)
plt.show()
print("Saved fig_opt_by_visa.png")

print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("Done.")
