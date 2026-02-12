# INITIAL DESCRIPTIVES OF IHMPs from IPEDS data 
# Imports and Paths
import duckdb as ddb
import pandas as pd
import sys
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import zipfile
import requests
try:
    import pgeocode
except Exception:  # pragma: no cover - optional dependency
    pgeocode = None
try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency
    gpd = None

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
#OUTFIG_PATH = '/Users/amykim/Documents/GitHub/h1bworkers/output/slides/slides_20251204'
OUTFIG_PATH = f"/home/yk0581/figures"
sns.set_theme(rc={"figure.figsize": (11, 6), "figure.dpi":200, "lines.linewidth": 3}, style="ticks", font_scale=1.4)
# Consistent IHMP styling across plots
IHMP_COLORS = {
    "IHMP": "#2e8b57",      # medium sea green
    "Non-IHMP": "#e07a5f",  # terracotta
}
# Shared shade family (variations on the Non-IHMP terracotta) so IHMP green pops
NON_IHMP_SHADES = {
    "All Degrees": "#bfc5d2",
    "Bachelor": "#4c78a8",
    "Master": "#e07a5f",
    "Non-IH Master": "#d99b83",
    "Doctor": "#2f2f38",
}
AWLEVEL_GROUP_ORDER = ["All Degrees", "Bachelor", "Master", "Doctor"]
AWLEVEL_GROUP_PALETTE = {
    "All Degrees": NON_IHMP_SHADES["All Degrees"],
    "Bachelor": NON_IHMP_SHADES["Bachelor"],
    "Master": NON_IHMP_SHADES["Master"],
    "Doctor": NON_IHMP_SHADES["Doctor"],
}
AWLEVEL_GROUP_MARKERS = {
    "All Degrees": "X",
    "Bachelor": "D",
    "Master": "s",
    "Doctor": "^",
}
# Neutral-forward palette for alternative award level grouping (IHMP stands out)
AWLEVEL_GROUP_ALT_ORDER = ["All Degrees", "Bachelor", "Non-IH Master", "IHMP", "Doctor"]
AWLEVEL_GROUP_ALT_PALETTE = {
    "IHMP": IHMP_COLORS["IHMP"],
    "Non-IH Master": NON_IHMP_SHADES["Non-IH Master"],
    "Bachelor": NON_IHMP_SHADES["Bachelor"],
    "Doctor": NON_IHMP_SHADES["Doctor"],
    "All Degrees": NON_IHMP_SHADES["All Degrees"],
}
AWLEVEL_GROUP_ALT_MARKERS = {
    "IHMP": "o",
    "Non-IH Master": "s",
    "Bachelor": "D",
    "Doctor": "^",
    "All Degrees": "X",
}
STEM_PALETTE = {
    "STEM": NON_IHMP_SHADES["Doctor"],
    "Non-STEM": NON_IHMP_SHADES["Bachelor"],
    "All Majors": NON_IHMP_SHADES["All Degrees"],
}
STEM_MARKERS = {
    "STEM": "o",
    "Non-STEM": "s",
    "All Majors": "X",
}


def flag_new_programs(df: pd.DataFrame, lookback_years: int = 5) -> pd.Series:
    """
    Flag program-year rows that are "new": no graduates for the same unitid x cipcode
    in the previous `lookback_years` years.
    """
    required = {"unitid", "cipcode", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for new program flag: {missing}")

    df_sorted = df.sort_values(["unitid", "cipcode", "year"]).copy()

    def mark(group: pd.DataFrame) -> pd.Series:
        years = group["year"].to_numpy()
        flags = []
        for idx, y in enumerate(years):
            prior = group["year"].iloc[:idx]
            has_recent = ((prior >= y - lookback_years) & (prior < y)).any()
            flags.append(0 if has_recent else 1)
        return pd.Series(flags, index=group.index)

    return df_sorted.groupby(["unitid", "cipcode"], group_keys=False).apply(mark)
#sns.set_context(font_scale = 2)

INT_FOLDER = f"{root}/data/int/int_files_nov2025"
IPEDS_GEO_PATH = f"{root}/data/raw/ipeds_cw_2021.csv"
# Optional shapefiles for county/ZIP outlines (set to your local paths)
ZIP_SHAPEFILE = os.environ.get("ZIP_SHAPEFILE_PATH", "")
COUNTY_SHAPEFILE = os.environ.get("COUNTY_SHAPEFILE_PATH", "")
ZCTA_CACHE_PATH = os.path.join(INT_FOLDER, "cb_2022_us_zcta520_500k.geojson")

# READING IN RAW DATA
ipeds = pd.read_parquet(f"{INT_FOLDER}/ipeds_completions_all.parquet")
ipeds['ihmp'] = (ipeds['share_intl'] >= 0.5)&(ipeds['awlevel'] == 7)&(ipeds['majornum'] == 1)&(ipeds['STEMOPT'] == 1)
ipeds['awlevel_group'] = np.where(ipeds['awlevel'].isin([5, 7, 17, 9]), ipeds['awlevel'].replace({
    5: 'Bachelor',
    7: 'Master',
    17: 'Doctor',
    9: 'Doctor'
}), 'Other')
ipeds['awlevel_group_alt'] = np.where((ipeds['awlevel']==7) & (ipeds['share_intl']>=0.5), 'IHMP', np.where(ipeds['awlevel']==7, 'Non-IH Master', ipeds['awlevel_group']))
ipeds['intl_ihmp'] = ipeds['ihmp']*ipeds['cnralt']
#ipeds['cip_cat'] = np.where(ipeds['cip2dig'].isin([11,14,27,40,52]))

# DESCRIPTIVE ONE: Share of all degree completions by international students over time
ipeds_by_year = ipeds[ipeds['awlevel_group'] != 'Other'].groupby(['year']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_deg_year = ipeds[ipeds['awlevel_group'] != 'Other'].groupby(['year', 'awlevel_group']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_deg_year_alt = ipeds[ipeds['awlevel_group_alt'] != 'Other'].groupby(['year', 'awlevel_group_alt']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_stem_year_ma = ipeds[(ipeds['awlevel_group'] == 'Master')].groupby(['year', 'STEMOPT']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_stem_year_all = ipeds[(ipeds['awlevel_group'] != 'Other')].groupby(['year', 'STEMOPT']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()

# normalizing to 2004 levels and adding share columns
dfs_out = []
normyear = 2004
for df in [ipeds_by_year, ipeds_by_deg_year, ipeds_by_stem_year_ma, ipeds_by_stem_year_all]:
    df['share_intl'] = df['cnralt'] / df['ctotalt']

    df_norm = df.copy()
    # normalizing to 2004 levels (by group)
    group_cols = [col for col in ['awlevel_group', 'STEMOPT'] if col in df_norm.columns]
    if group_cols:
        df_norm[['ctotalt', 'cnralt', 'intl_ihmp', 'share_intl']] = df_norm.groupby(group_cols)[['ctotalt', 'cnralt', 'intl_ihmp', 'share_intl']].transform(lambda x: x / x[df_norm['year'] == normyear].values[0])
    else:
        df_norm[['ctotalt', 'cnralt', 'intl_ihmp', 'share_intl']] = df_norm[['ctotalt', 'cnralt', 'intl_ihmp', 'share_intl']].div(df_norm.loc[df_norm['year'] == normyear, ['ctotalt', 'cnralt', 'intl_ihmp', 'share_intl']].values[0])

    dfs_out.append((df, df_norm))

# PLOT 1a: Share of international students over time (all degrees)
g = sns.lineplot(data=dfs_out[0][0], x='year', y='share_intl', color=NON_IHMP_SHADES["All Degrees"], marker="o")
g.set_xlabel("Graduation Year")
g.set_ylabel("International Bachelor+ Graduates as Share of Total")
g.figure.savefig(f"{OUTFIG_PATH}/ipeds_share_intl_over_time_all_degrees.png")

print(f"Increase in share international (all degrees) from 2004 to 2024: {round(dfs_out[0][0].loc[dfs_out[0][0]['year']==2024, 'share_intl'].values[0]/dfs_out[0][0].loc[dfs_out[0][0]['year']==2004, 'share_intl'].values[0],2)}x")

# PLOT 1b: Number of international students over time (all degrees)
df_1b = dfs_out[0][0].copy()
df_1b['cnralt'] = df_1b['cnralt']/1000  # rescaling
plt.figure()
g2 = sns.lineplot(data=df_1b, x='year', y='cnralt', color=NON_IHMP_SHADES["All Degrees"], marker="o")
g2.set_xlabel("Graduation Year")
g2.set_ylabel("International Bachelor+ Graduates (thousands)")
g2.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_intl_over_time_all_degrees.png")
    
print(f"Increase in international graduates (all degrees) from 2004 to 2024: {round(dfs_out[0][0].loc[dfs_out[0][0]['year']==2024, 'cnralt'].values[0]/dfs_out[0][0].loc[dfs_out[0][0]['year']==2004, 'cnralt'].values[0],2)}x")

# PLOT 2a: Share of international students over time (by degree)
df_2a = pd.concat([dfs_out[1][0].copy(), pd.DataFrame({'year': dfs_out[0][0]['year'], 'awlevel_group': 'All Degrees', 'share_intl': dfs_out[0][0]['share_intl']})], ignore_index=True)
plt.figure()
g3 = sns.lineplot(
    data=df_2a,
    x='year',
    y='share_intl',
    hue='awlevel_group',
    style='awlevel_group',
    hue_order=[g for g in AWLEVEL_GROUP_ORDER if g in df_2a['awlevel_group'].unique()],
    style_order=[g for g in AWLEVEL_GROUP_ORDER if g in df_2a['awlevel_group'].unique()],
    palette=AWLEVEL_GROUP_PALETTE,
    markers=AWLEVEL_GROUP_MARKERS,
    dashes=False,
)
g3.set_xlabel("Graduation Year")
g3.set_ylabel("International Graduates as Share of Total")
g3.legend(title='Degree Level', loc = 'upper left')
g3.figure.savefig(f"{OUTFIG_PATH}/ipeds_share_intl_over_time_by_degree.png")
# PLOT 2b: Number of international students over time (by degree + all degrees)
df_2b = pd.concat([dfs_out[1][0].copy(), pd.DataFrame({'year': dfs_out[0][0]['year'], 'awlevel_group': 'All Degrees', 'cnralt': dfs_out[0][0]['cnralt']})], ignore_index=True)
df_2b['cnralt'] = df_2b['cnralt']/1000  # rescaling
plt.figure()
g4 = sns.lineplot(
    data=df_2b,
    x='year',
    y='cnralt',
    hue='awlevel_group',
    style='awlevel_group',
    hue_order=[g for g in AWLEVEL_GROUP_ORDER if g in df_2b['awlevel_group'].unique()],
    style_order=[g for g in AWLEVEL_GROUP_ORDER if g in df_2b['awlevel_group'].unique()],
    palette=AWLEVEL_GROUP_PALETTE,
    markers=AWLEVEL_GROUP_MARKERS,
    dashes=False,
)
g4.set_xlabel("Graduation Year")
g4.set_ylabel("International Graduates (thousands)")
g4.legend(title='Degree Level', loc = 'upper left')
g4.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_intl_over_time_by_degree.png")

# PLOT 2b (alt): Number of international students over time using awlevel_group_alt (exclude All Degrees)
df_2b_alt = ipeds_by_deg_year_alt.copy()
df_2b_alt = df_2b_alt[df_2b_alt['awlevel_group_alt'] != 'Other']
df_2b_alt['cnralt'] = df_2b_alt['cnralt'] / 1000
plt.figure()
g4_alt = sns.lineplot(
    data=df_2b_alt,
    x='year',
    y='cnralt',
    hue='awlevel_group_alt',
    style='awlevel_group_alt',
    hue_order=[g for g in AWLEVEL_GROUP_ALT_ORDER if g in df_2b_alt['awlevel_group_alt'].unique()],
    style_order=[g for g in AWLEVEL_GROUP_ALT_ORDER if g in df_2b_alt['awlevel_group_alt'].unique()],
    palette=AWLEVEL_GROUP_ALT_PALETTE,
    markers=AWLEVEL_GROUP_ALT_MARKERS,
    dashes=False,
)
g4_alt.set_xlabel("Graduation Year")
g4_alt.set_ylabel("International Graduates (thousands)")
g4_alt.legend(title='Degree Level', loc='upper left')
g4_alt.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_intl_over_time_by_degree_alt.png")

# print share of all intl degrees from ma in 2024
print(f"Share of all int'l students in MA in 2024: {round(df_2b.loc[(df_2b['year']==2024)&(df_2b['awlevel_group']=='Master'), 'cnralt'].values[0]/df_2b.loc[(df_2b['year']==2024)&(df_2b['awlevel_group']=='All Degrees'), 'cnralt'].values[0],2)}")

# print share of all total degrees from ma in 2024
ipeds_by_deg_year_tot = ipeds[ipeds['awlevel_group'] != 'Other'].groupby(['year', 'awlevel_group']).agg({'ctotalt': 'sum'}).reset_index()
print(f"Share of all students in MA in 2024: {round(ipeds_by_deg_year_tot.loc[(ipeds_by_deg_year_tot['year']==2024)&(ipeds_by_deg_year_tot['awlevel_group']=='Master'), 'ctotalt'].values[0]/ipeds_by_deg_year_tot.loc[(ipeds_by_deg_year_tot['year']==2024), 'ctotalt'].sum(),2)}")

# PLOT 3a: Share of international students over time (by STEMOPT status)
for masters in [True, False]:
    if masters:
        # stemopt for master's only
        stemopt_df_3 = dfs_out[2][0].copy()
        # all degrees for master's only
        all_deg_df_3 = dfs_out[1][0][dfs_out[1][0]['awlevel_group']=='Master'] # only masters
        suff = "_ma"
        ylab_addl = " (Master's Only)"
    else:
        stemopt_df_3 = dfs_out[3][0].copy()
        all_deg_df_3 = dfs_out[0][0] # all
        suff = "_all"
        ylab_addl = ""

    stemopt_df_3['STEMOPT'] = np.where(stemopt_df_3['STEMOPT']==1, 'STEM', 'Non-STEM')
    df_3 = pd.concat([stemopt_df_3, pd.DataFrame({'year': all_deg_df_3['year'], 'STEMOPT': 'All Majors', 'share_intl': all_deg_df_3['share_intl'],'cnralt': all_deg_df_3['cnralt']})], ignore_index=True)
    plt.figure()
    g4 = sns.lineplot(
        data=df_3,
        x='year',
        y='share_intl',
        hue='STEMOPT',
        style='STEMOPT',
        palette=STEM_PALETTE,
        markers=STEM_MARKERS,
        dashes=False,
    )
    g4.set_xlabel("Graduation Year")
    g4.set_ylabel("International Graduates as Share of Total" + ylab_addl)
    g4.legend(title='Major Type', loc = 'upper left')
    g4.figure.savefig(f"{OUTFIG_PATH}/ipeds_share_intl_over_time_by_stemopt{suff}.png")

    # PLOT 3b: Number of international students over time (by STEMOPT status)
    df_3['cnralt'] = df_3['cnralt']/1000  # rescaling
    plt.figure()
    g5 = sns.lineplot(
        data=df_3,
        x='year',
        y='cnralt',
        hue='STEMOPT',
        style='STEMOPT',
        palette=STEM_PALETTE,
        markers=STEM_MARKERS,
        dashes=False,
    )
    g5.set_xlabel("Graduation Year")
    g5.set_ylabel("International Graduates (thousands)" + ylab_addl)
    g5.legend(title='Major Type', loc = 'upper left')
    g5.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_intl_over_time_by_stemopt{suff}.png")

# # PLOT 4a: Share of international students by degree type over time
# df_4a = dfs_out[1][0].copy()
# df_4a['share_of_intl'] = df_4a.groupby('year')['cnralt'].transform(lambda x: x / x.sum())
# g6 = sns.lineplot(data=df_4a, x='year', y='share_of_intl', hue='awlevel_group')
# g6.set_xlabel("Graduation Year")

# DESCRIPTIVE TWO: Number of programs over time by share_intl decile
ncuts = 2
#groupby = ['year']
groupby = ['year', 'awlevel_group']
ipeds_by_share_decile_by_year = ipeds[(ipeds['awlevel_group'] != 'Other')&(ipeds['ctotalt']>=10)].copy()
ipeds_by_share_decile_by_year['share_intl_decile'] = pd.cut(ipeds_by_share_decile_by_year['share_intl'], bins=[i/ncuts for i in range(ncuts + 1)], labels=[f"{round(100*i/ncuts)}-{round(100*(i+1)/ncuts)}%" for i in range(ncuts)], include_lowest=True)
ipeds_by_share_decile_by_year = ipeds_by_share_decile_by_year.groupby(groupby + ['share_intl_decile']).agg({'unitid': 'nunique', 'cnralt': 'sum', 'ctotalt': 'mean'}).reset_index().rename(columns={'unitid': 'n_programs', 'ctotalt': 'avg_program_size'})  
ipeds_by_share_decile_by_year['share_programs'] = ipeds_by_share_decile_by_year.groupby(groupby)['n_programs'].transform(lambda x: x / x.sum())
ipeds_by_share_decile_by_year['share_students'] = ipeds_by_share_decile_by_year.groupby(groupby)['cnralt'].transform(lambda x: x / x.sum())

# normalize to 2004 levels
group_cols = ['share_intl_decile', 'awlevel_group']
normyear = 2004

# 1. Get 2004 baseline per group
base_2004 = (
    ipeds_by_share_decile_by_year
    [ipeds_by_share_decile_by_year['year'] == normyear]
    .set_index(group_cols)['n_programs']
    .rename('n_programs_2004')
)

# 2. Merge back into full df
ipeds_by_share_decile_by_year = ipeds_by_share_decile_by_year.merge(
    base_2004,
    on=group_cols,
    how='left'
)

# 3. Normalized series (relative to 2004 level)
ipeds_by_share_decile_by_year['n_programs_norm'] = (
    ipeds_by_share_decile_by_year['n_programs'] /
    ipeds_by_share_decile_by_year['n_programs_2004']
)

# 4. Label using the actual 2004 level, not the normalized value
ipeds_by_share_decile_by_year['share_intl_decile_label'] = (
    ipeds_by_share_decile_by_year
    .apply(
        lambda row: f"{row['share_intl_decile']} (2004: {int(row['n_programs_2004'])})",
        axis=1
    )
    )

share_decile_palette = sns.light_palette(NON_IHMP_SHADES["Master"], n_colors=ncuts, reverse=True)
share_decile_palette = [
    IHMP_COLORS["Non-IHMP"],
    IHMP_COLORS["IHMP"],
][:ncuts]

# PLOT 1: Number of programs over time by share_intl decile
plt.figure()
g2_1 = sns.lineplot(
    data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')],
    x='year',
    y='n_programs_norm',
    hue='share_intl_decile_label',
    style='share_intl_decile_label',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_1.set_xlabel("Graduation Year")
g2_1.set_ylabel("Number of Programs (normalized to 2004)")
g2_1.legend(title="Program Pct Int'l (2004 # Programs)", loc = 'upper left')
g2_1.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_programs_by_share_intl_over_time_master_norm.png")
# print number of new programs between 2004 and 2024 for each decile
for decile in ipeds_by_share_decile_by_year['share_intl_decile'].unique():
    n_programs_2004 = ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['year']==2004)&(ipeds_by_share_decile_by_year['awlevel_group']=='Master')&(ipeds_by_share_decile_by_year['share_intl_decile']==decile)]['n_programs'].values[0]
    n_programs_2024 = ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['year']==2024)&(ipeds_by_share_decile_by_year['awlevel_group']=='Master')&(ipeds_by_share_decile_by_year['share_intl_decile']==decile)]['n_programs'].values[0]
    print(f"Decile {decile}: New programs from 2004 to 2024: {n_programs_2024 - n_programs_2004} (from {n_programs_2004} to {n_programs_2024})")

# PLOT 1b: Number of programs (not normalized)
plt.figure()
g2_1b = sns.lineplot(
    data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')],
    x='year',
    y='n_programs',
    hue='share_intl_decile_label',
    style='share_intl_decile_label',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_1b.set_xlabel("Graduation Year")
g2_1b.set_ylabel("Number of Master's Programs")
g2_1b.legend(title="Program Pct Int'l (2004 # Programs)", loc = 'upper left')
g2_1b.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_programs_by_share_intl_over_time_stem_master.png")

# PLOT 1b-new: Number of new programs (no grads in prior 5 years) by share_intl decile
prog_decile_base = ipeds[(ipeds['awlevel_group'] != 'Other') & (ipeds['ctotalt'] >= 10)].copy()
prog_decile_base['share_intl_decile'] = pd.cut(
    prog_decile_base['share_intl'],
    bins=[i / ncuts for i in range(ncuts + 1)],
    labels=[f"{round(100 * i / ncuts)}-{round(100 * (i + 1) / ncuts)}%" for i in range(ncuts)],
    include_lowest=True,
)
prog_decile_base = prog_decile_base.dropna(subset=['share_intl_decile'])
prog_decile_base['is_new_program'] = flag_new_programs(prog_decile_base, lookback_years=5)
new_programs_by_decile = (
    prog_decile_base.groupby(['year', 'awlevel_group', 'share_intl_decile'], as_index=False)
    .agg(new_programs=('is_new_program', 'sum'))
)

plt.figure()
g2_1b_new = sns.lineplot(
    data=new_programs_by_decile[(new_programs_by_decile['awlevel_group'] == 'Master')&(new_programs_by_decile['year'].between(2006, 2023))],
    x='year',
    y='new_programs',
    hue='share_intl_decile',
    style='share_intl_decile',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_1b_new.set_xlabel("Graduation Year")
g2_1b_new.set_ylabel("New Master's Programs (no grads in prior 5y)")
g2_1b_new.legend(title="Program Pct Int'l", loc='upper left')
g2_1b_new.figure.savefig(f"{OUTFIG_PATH}/ipeds_new_programs_by_share_intl_over_time_master.png")

# PLOT 1b-new-normalized: New programs normalized to 2008 levels
normyear_new = 2008
new_prog_master = new_programs_by_decile[new_programs_by_decile['awlevel_group'] == 'Master'].copy()
base_new = (
    new_prog_master[new_prog_master['year'] == normyear_new]
    .set_index('share_intl_decile')['new_programs']
    .rename('new_programs_2008')
)
new_prog_master = new_prog_master.merge(base_new, on='share_intl_decile', how='left')
new_prog_master['new_programs_norm'] = new_prog_master['new_programs'] / new_prog_master['new_programs_2008']

plt.figure()
g2_1b_new_norm = sns.lineplot(
    data=new_prog_master[new_prog_master['year'].between(2006, 2023)],
    x='year',
    y='new_programs_norm',
    hue='share_intl_decile',
    style='share_intl_decile',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_1b_new_norm.set_xlabel("Graduation Year")
g2_1b_new_norm.set_ylabel(f"New Master's Programs (normalized to {normyear_new})")
g2_1b_new_norm.legend(title="Program Pct Int'l", loc='upper left')
g2_1b_new_norm.figure.savefig(f"{OUTFIG_PATH}/ipeds_new_programs_by_share_intl_over_time_master_norm{normyear_new}.png")

# PLOT 1c: Share of programs over time by share_intl decile
plt.figure()
g2_1c = sns.lineplot(
    data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')],
    x='year',
    y='share_programs',
    hue='share_intl_decile',
    style='share_intl_decile',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_1c.set_xlabel("Graduation Year")
g2_1c.set_ylabel("Share of Master's Programs")
g2_1c.legend(title="Program Pct Int'l", loc = 'upper left')
g2_1c.figure.savefig(f"{OUTFIG_PATH}/ipeds_share_programs_by_share_intl_over_time_stem_master.png")

# PLOT 2: Share of students over time by share_intl decile
plt.figure()
#g2_2 = sns.lineplot(data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')], x='year', y='share_students', hue='share_intl_decile', style = 'share_intl_decile', palette=share_decile_palette)

g2_2 = sns.lineplot(
    data=ipeds_by_share_decile_by_year,
    x='year',
    y='share_students',
    hue='share_intl_decile',
    style='share_intl_decile',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_2.set_xlabel("Graduation Year")
g2_2.set_ylabel("Share of International Students")
g2_2.legend(title="Program Pct Int'l", loc = 'upper left')
g2_2.figure.savefig(f"{OUTFIG_PATH}/ipeds_share_students_by_share_intl_over_time_all.png")
# print increase in share of students between 2004 and 2024 for each decile
for decile in ipeds_by_share_decile_by_year['share_intl_decile'].unique():
    share_students_2004 = ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['year']==2004)&(ipeds_by_share_decile_by_year['share_intl_decile']==decile)]['share_students'].values[0]
    share_students_2024 = ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['year']==2024)&(ipeds_by_share_decile_by_year['share_intl_decile']==decile)]['share_students'].values[0]
    print(f"Decile {decile}: Increase in share of students from 2004 to 2024: {round(share_students_2024/share_students_2004,2)}x (from {round(share_students_2004,3)} to {round(share_students_2024,3)})")

# PLOT 2b: Number of students over time by share_intl decile
plt.figure()
g2_2b = sns.lineplot(
    data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')],
    x='year',
    y='cnralt',
    hue='share_intl_decile',
    style='share_intl_decile',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g2_2b.set_xlabel("Graduation Year")
g2_2b.set_ylabel("Number of Int'l MA STEM Students")
g2_2b.legend(title="Program Pct Int'l", loc = 'upper left')
g2_2b.figure.savefig(f"{OUTFIG_PATH}/ipeds_n_students_by_share_intl_over_time_stem_master.png")

# PLOT 3: Average program size over time by share_intl decile
plt.figure()
g3 = sns.lineplot(
    data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')],
    x='year',
    y='avg_program_size',
    hue='share_intl_decile',
    style='share_intl_decile',
    palette=share_decile_palette,
    markers=True,
    dashes=False,
)
g3.set_xlabel("Graduation Year")
g3.set_ylabel("Average Program Size")
g3.legend(title="Program Pct Int'l", loc = 'upper left')
g3.figure.savefig(f"{OUTFIG_PATH}/ipeds_avg_program_size_by_share_intl_over_time_stem_master.png")

# DESCRIPTIVE THREE: Tracking programs over time
# DATA PREP: Group by UNITID x CIPCODE x STEMOPT for MA only, get 2004 and 2024 totals and intl students and intl shares
ipeds_ma = ipeds[ipeds['awlevel_group']=='Master']
ipeds_2004 = ipeds_ma[ipeds_ma['year']==2004][['unitid', 'cipcode', 'ctotalt', 'cnralt']].rename(columns={'ctotalt': 'ctotalt_2004', 'cnralt': 'cnralt_2004'})
ipeds_2024 = ipeds_ma[ipeds_ma['year']==2024][['unitid', 'cipcode', 'ctotalt', 'cnralt']].rename(columns={'ctotalt': 'ctotalt_2024', 'cnralt': 'cnralt_2024'})
ipeds_2004_2024 = ipeds_2004.merge(ipeds_2024, on=['unitid', 'cipcode'], how='inner')
# fill nas with 0
ipeds_2004_2024['ctotalt_2004'] = ipeds_2004_2024['ctotalt_2004'].fillna(0)
ipeds_2004_2024['cnralt_2004'] = ipeds_2004_2024['cnralt_2004'].fillna(0)
ipeds_2004_2024['ctotalt_2024'] = ipeds_2004_2024['ctotalt_2024'].fillna(0)
ipeds_2004_2024['cnralt_2024'] = ipeds_2004_2024['cnralt_2024'].fillna(0)
ipeds_2004_2024['share_intl_2004'] = np.where(ipeds_2004_2024['ctotalt_2004']>0, ipeds_2004_2024['cnralt_2004']/ipeds_2004_2024['ctotalt_2004'], 0)
ipeds_2004_2024['share_intl_2024'] = np.where(ipeds_2004_2024['ctotalt_2024']>0, ipeds_2004_2024['cnralt_2024']/ipeds_2004_2024['ctotalt_2024'], 0)
# compute 2004 to 2024 growth in total students and intl students
ipeds_2004_2024['growth_total'] = np.where(ipeds_2004_2024['ctotalt_2004']>0, (ipeds_2004_2024['ctotalt_2024']-ipeds_2004_2024['ctotalt_2004'])/ipeds_2004_2024['ctotalt_2004'], np.nan)
ipeds_2004_2024['growth_intl'] = np.where(ipeds_2004_2024['cnralt_2004']>0, (ipeds_2004_2024['cnralt_2024']-ipeds_2004_2024['cnralt_2004'])/ipeds_2004_2024['cnralt_2004'], np.nan)

# PLOT 1: Distribution of growth rates for total 
plt.figure()
g3_1 = sns.histplot(data=ipeds_2004_2024[ipeds_2004_2024['growth_total']<10], x='growth_total', bins=50, kde=False)

# PLOT 2: share intl in 2004 vs share intl in 2024 (for those with at least 10 students in both years)
plt.figure()
g3_2 = sns.regplot(data=ipeds_2004_2024[(ipeds_2004_2024['ctotalt_2004']>=10) & (ipeds_2004_2024['ctotalt_2024']>=10)], x='share_intl_2004', y='share_intl_2024', x_bins=20, fit_reg = False)

plt.plot([0, 1], [0, 1], color='red', linestyle='--')

# PLOT 3: distribution of change in share intl from 2004 to 2024
ipeds_2004_2024['share_intl_change'] = ipeds_2004_2024['share_intl_2024'] - ipeds_2004_2024['share_intl_2004']
plt.figure()
g3_3 = sns.histplot(data=ipeds_2004_2024[np.abs(ipeds_2004_2024['share_intl_change'])<0.5], x='share_intl_change', bins=100, kde=False)


# PLOT 1: Each UNITID x CIPCODE is a dot, x axis is 2004 share international, 
# subset: only those that exist in both years
ipeds_ma = ipeds[ipeds['awlevel_group']=='Master']
ipeds_2004 = ipeds_ma[ipeds_ma['year']==2004][['unitid', 'cipcode', 'ctotalt']].rename(columns={'ctotalt': 'ctotalt_2004'})
ipeds_2024 = ipeds_ma[ipeds_ma['year']==2024][['unitid', 'cipcode', 'ctotalt']].rename(columns={'ctotalt': 'ctotalt_2024'})
ipeds_2004_2024 = ipeds_2004.merge(ipeds_2024, on=['unitid', 'cipcode'], how='inner')
# fill nas with 0
ipeds_2004_2024['ctotalt_2004'] = ipeds_2004_2024['ctotalt_2004'].fillna(0)
ipeds_2004_2024['ctotalt_2024'] = ipeds_2004_2024['ctotalt_2024'].fillna(0)



# get rid of outliers
maxn = 500
ipeds_2004_2024 = ipeds_2004_2024[(ipeds_2004_2024['ctotalt_2004']<=maxn)&(ipeds_2004_2024['ctotalt_2024']<=maxn)]
plt.figure()
g3_1 = sns.regplot(data=ipeds_2004_2024, x='ctotalt_2004', y='ctotalt_2024', x_bins=50, fit_reg = False)
# add 45 degree line
plt.plot([0, 200], [0, 200], color='red', linestyle='--')

# PLOT 2: Same as above, but color by quartile of share_intl in 2004
ipeds_2004_full = ipeds[ipeds['year']==2004][['unitid', 'cipcode', 'ctotalt', 'cnralt']].rename(columns={'ctotalt': 'ctotalt_2004', 'cnralt': 'cnralt_2004'})

# DESCRIPTIVE THREE: Number/share of programs/students over time by school characteristics
# PLOT 1: Share of MA students over time by instsize_lab
df_size = ipeds[(ipeds['awlevel_group']=='Master')&(ipeds['ctotalt']>=10)].copy()
df_size = df_size.groupby(['year', 'instsize_lab']).agg({'unitid': 'nunique', 'cnralt': 'sum'}).reset_index().rename(columns={'unitid': 'n_programs'})
df_size['share_programs'] = df_size.groupby(['year'])['n_programs'].transform(lambda x: x / x.sum())
df_size['share_students'] = df_size.groupby(['year'])['cnralt'].transform(lambda x: x / x.sum())
plt.figure()
g3_1 = sns.lineplot(data=df_size, x='year', y='cnralt', hue='instsize_lab')

# PLOT 2: Share of MA students over time by instcat_lab
df_cat = ipeds[(ipeds['awlevel_group']=='Master')&(ipeds['ctotalt']>=10)].copy()

# color palette gradient by decile
g2_1 = sns.lineplot(data=ipeds_by_share_decile_by_year[(ipeds_by_share_decile_by_year['awlevel_group']=='Master')], x='year', y='share_programs', hue='share_intl_decile', palette=cmap)

norm = 0
# keeping plot y axis consistent for comparison
ymax = dfs_out[1][norm]['share_intl'].max()*1.1
ymin = dfs_out[1][norm]['share_intl'].min()*0.9

sns.lineplot(data=dfs_out[0][norm], x='year', y='share_intl', color = 'gray', ylim=(ymin, ymax))
sns.lineplot(data=dfs_out[1][norm], x='year', y='share_intl', hue='awlevel_group', ylim=(ymin, ymax))

norm = 0
sns.lineplot(data=dfs_out[0][norm], x='year', y='cnralt', color = 'gray')
sns.lineplot(data=dfs_out[1][norm], x='year', y='cnralt', hue='awlevel_group')


sns.lineplot(data=ipeds_by_year, x='year', y='share_intl', color = 'gray')


sns.lineplot(data=ipeds_by_year, x='year', y='ctotalt', color = 'gray')

sns.lineplot(data=ipeds_by_year, x='year', y='share_intl', color = 'gray')
sns.lineplot(data=ipeds_by_deg_year[ipeds_by_deg_year['awlevel_group']=='Bachelor'], x='year', y='share_intl', hue='awlevel_group')

sns.lineplot(data=ipeds_by_year, x='year', y='cnralt', color = 'gray')
sns.lineplot(data=ipeds_by_deg_year, x='year', y='cnralt', hue='awlevel_group')

# DESCRIPTIVE TWO: Share of Master's IHMPs over time
ipeds_masters = ipeds[ipeds['awlevel'] == 7]
ipeds_masters_by_year = ipeds_masters.groupby(['year']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
# ipeds_masters_by_year['share_intl'] = ipeds_masters_by_year['cnralt'] / ipeds_masters_by_year['ctotalt

# DESCRIPTIVE THREE: Spatial distribution of IHMPs over time
def _resolve_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_zcta_shapes() -> "gpd.GeoDataFrame | None":
    """
    Load ZCTA boundaries via Census cartographic files.

    Tries, in order:
    1) ZIP_SHAPEFILE env var (if provided)
    2) Cached GeoJSON at ZCTA_CACHE_PATH
    3) Download Census cartographic ZCTA (cb_2022_us_zcta520_500k.zip)
    """

    if not gpd:
        return None

    if ZIP_SHAPEFILE and os.path.exists(ZIP_SHAPEFILE):
        try:
            return gpd.read_file(ZIP_SHAPEFILE)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read ZIP_SHAPEFILE {ZIP_SHAPEFILE} ({exc}); continuing.")

    if os.path.exists(ZCTA_CACHE_PATH):
        try:
            return gpd.read_file(ZCTA_CACHE_PATH)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read cached ZCTA geojson at {ZCTA_CACHE_PATH} ({exc}); re-downloading.")

    url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_zcta520_500k.zip"
    try:
        print("Downloading ZCTA boundaries from Census (cb_2022_us_zcta520_500k)...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(INT_FOLDER)
        shp_path = os.path.join(INT_FOLDER, "cb_2022_us_zcta520_500k.shp")
        gdf = gpd.read_file(shp_path)
        try:
            gdf.to_file(ZCTA_CACHE_PATH, driver="GeoJSON")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to cache ZCTA geojson ({exc})")
        return gdf
    except Exception as exc:  # pragma: no cover
        print(f"Failed to download Census ZCTA boundaries ({exc}); falling back to point plot.")
        return None


if os.path.exists(IPEDS_GEO_PATH):
    ipeds_geo = pd.read_csv(IPEDS_GEO_PATH)
    lat_col = _resolve_col(ipeds_geo, ["lat", "latitude", "lat_dd"])
    lon_col = _resolve_col(ipeds_geo, ["lon", "lng", "longitude", "long_dd"])
    zip_col = _resolve_col(ipeds_geo, ["zip", "zip_code", "postalcode", "postal_code"])
    unit_col = _resolve_col(ipeds_geo, ["unitid", "UNITID"])
    state_col = _resolve_col(ipeds_geo, ["stabbr", "state", "state_abbr"])

    # If lat/lon missing, try deriving them from ZIP codes via pgeocode (if available).
    if zip_col and (not lat_col or not lon_col):
        if pgeocode is None:
            print("Skipping spatial IHMP plot: no lat/lon columns and pgeocode not installed to derive them from ZIP.")
        else:
            # Normalize ZIPs to 5-digit strings
            zip_norm_col = "zip_norm_tmp"
            ipeds_geo[zip_norm_col] = ipeds_geo[zip_col].astype(str).str.slice(0, 5).str.zfill(5)
            nomi = pgeocode.Nominatim("us")
            zip_lookup = ipeds_geo[zip_norm_col].dropna().unique()
            zip_df = pd.DataFrame({"zip_norm": zip_lookup})
            zip_df["lat_dd"] = zip_df["zip_norm"].apply(lambda z: nomi.query_postal_code(z).latitude)
            zip_df["long_dd"] = zip_df["zip_norm"].apply(lambda z: nomi.query_postal_code(z).longitude)
            zip_df["state_code"] = zip_df["zip_norm"].apply(lambda z: nomi.query_postal_code(z).state_code)
            ipeds_geo = ipeds_geo.merge(
                zip_df,
                left_on=zip_norm_col,
                right_on="zip_norm",
                how="left",
            )
            lat_col = "lat_dd"
            lon_col = "long_dd"
            if state_col is None and "state_code" in ipeds_geo.columns:
                state_col = "state_code"
            if zip_col is None:
                zip_col = zip_norm_col

    if lat_col and lon_col and unit_col and zip_col:
        ihmp_geo = (
            ipeds_masters.loc[(ipeds_masters["ihmp"]) & (ipeds_masters["ctotalt"] >= 10)]
            .merge(
                ipeds_geo[[col for col in [unit_col, lat_col, lon_col, zip_col, state_col] if col]],
                left_on="unitid",
                right_on=unit_col,
                how="left",
            )
        )
        if state_col:
            ihmp_geo = ihmp_geo[~ihmp_geo[state_col].isin(["AK", "HI"])]

        # total unitids per zip for denominator
        zip_totals = (
            ipeds_geo.groupby(zip_col, as_index=False)[unit_col].nunique().rename(columns={unit_col: "total_unitids"})
        )

        ihmp_geo_agg = (
            ihmp_geo.groupby(["year", zip_col], as_index=False)
            .agg(
                ihmp_unitids=("unitid", "nunique"),
                lat=(lat_col, "first"),
                lon=(lon_col, "first"),
                state=(state_col, "first") if state_col else ("unitid", "size"),
            )
            .merge(zip_totals, on=zip_col, how="left")
            .rename(columns={"lon": "plot_lon", "lat": "plot_lat", zip_col: "zip_code"})
        )
        ihmp_geo_agg["ihmp_share"] = ihmp_geo_agg["ihmp_unitids"] / ihmp_geo_agg["total_unitids"]

        shapes = _load_zcta_shapes() if gpd else None
        if gpd and shapes is not None:
            geo_key = _resolve_col(shapes, ["GEOID", "ZCTA5CE10", "ZCTA5CE20", "ZIP", "ZIPCODE", "ZIP_CODE", "ZCTA"])
            if geo_key:
                shapes = shapes.rename(columns={geo_key: "zip_code"})
                shapes["zip_code"] = shapes["zip_code"].astype(str).str.extract(r"(\\d{3,5})")[0].str.zfill(5)
                merged = shapes.merge(ihmp_geo_agg, on="zip_code", how="left")
                if state_col and "state" in merged.columns:
                    merged = merged[~merged["state"].isin(["AK", "HI"])]
                years = sorted(merged["year"].dropna().unique())
                ncols = 4
                nrows = int(np.ceil(len(years) / ncols)) or 1
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes = np.atleast_1d(axes).flatten()
                for ax, yr in zip(axes, years):
                    merged[merged["year"] == yr].plot(
                        column="ihmp_share",
                        cmap="viridis",
                        linewidth=0.05,
                        edgecolor="gray",
                        legend=False,
                        ax=ax,
                    )
                    ax.set_title(f"Year {yr}")
                    ax.set_axis_off()
                for ax in axes[len(years) :]:
                    ax.set_axis_off()
                fig.suptitle("Spatial distribution of IHMP programs (ZIP polygons, ctotalt>=10, excluding AK/HI)", y=0.92)
                plt.tight_layout()
            else:
                print("Could not resolve ZIP field in Census shapes; falling back to point plot.")
                shapes = None
        if not gpd or shapes is None:
            g = sns.relplot(
                data=ihmp_geo_agg,
                x="plot_lon",
                y="plot_lat",
                size="ihmp_unitids",
                hue="ihmp_share",
                col="year",
                col_wrap=4,
                height=3,
                palette="viridis",
                sizes=(20, 200),
            )
            g.set_titles("Year {col_name}")
            g.set_axis_labels("Longitude", "Latitude")
            plt.suptitle("Spatial distribution of IHMP programs (ZIP aggregated, ctotalt>=10, excluding AK/HI)", y=1.02)
            plt.tight_layout()
    else:
        print("Skipping spatial IHMP plot: could not resolve lat/lon/zip/unitid columns in crosswalk.")
else:
    print(f"Skipping spatial IHMP plot: geoname crosswalk not found at {IPEDS_GEO_PATH}.")

# DESCRIPTIVE FOUR: Econ field composition over time (relative to 2010)
def _pad_cip(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: str(int(x)).zfill(6) if pd.notna(x) else None)


ipeds["cip_str"] = _pad_cip(ipeds.get("cipcode", pd.Series(dtype="Int64")))

def _assign_econ_group(cip: str) -> str | None:
    if cip is None:
        return None
    if cip == "450601":
        return "General Economics (Non-STEM OPT)"
    if cip == "450603":
        return "Econometrics (STEM OPT)"
    if cip in {"450602", "450604", "450605", "450699"}:
        return "Other Economics (Non-STEM OPT)"
    return None


def _assign_fin_math_group(cip: str) -> str | None:
    if cip is None:
        return None
    if cip.startswith("5208"):
        return "Finance (Non-STEM OPT)"
    if cip.startswith("2701"):
        return "General Math (STEM OPT)"
    if cip.startswith("2703"):
        return "Applied Math (STEM OPT)"
    return None


# Econ comparison
econ = ipeds[ipeds["awlevel_group"] != "Other"].copy()
econ["econ_group"] = econ["cip_str"].apply(_assign_econ_group)
econ = econ[econ["econ_group"].notna()]
econ_grouped = (
    econ.groupby(["awlevel_group", "year", "econ_group"], as_index=False)["ctotalt"]
    .sum()
    .rename(columns={"ctotalt": "completions"})
)
baseline_econ = econ_grouped[econ_grouped["year"] == 2010][["awlevel_group", "econ_group", "completions"]].rename(
    columns={"completions": "base_2010"}
)
econ_grouped = econ_grouped.merge(baseline_econ, on=["awlevel_group", "econ_group"], how="left")
econ_grouped["rel_to_2010"] = econ_grouped["completions"] / econ_grouped["base_2010"]
econ_grouped = econ_grouped[econ_grouped['awlevel_group']=="Master"]

fig, axes = plt.subplots(1, len(econ_grouped["awlevel_group"].unique()), figsize=(11, 6), sharey=True)
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
for ax, degree in zip(axes, sorted(econ_grouped["awlevel_group"].unique())):
    subset = econ_grouped[econ_grouped["awlevel_group"] == degree]
    for group_name, grp in subset.groupby("econ_group"):
        ax.plot(grp["year"], grp["completions"], marker="o", label = group_name)
    ax.axvline(2014, color = 'red', linestyle='--', linewidth=1)
    # add grey box between 2014 and 2016
    ax.axvspan(2014, 2016, color='gray', alpha=0.3)
    ax.axvline(2016, color = 'red', linestyle='--', linewidth=1)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Total Master's Degrees")
    ax.set_xlabel("Year")
    ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTFIG_PATH}/ipeds_ma_econ_completions_by_year.png")

# Finance / Math / Statistics comparison
tabfield = 'ctotalt'  # can switch to 'ctotalt' for total completions instead of intl
fin_math = ipeds[ipeds["awlevel_group"] != "Other"].copy()
fin_math["field_group"] = fin_math["cip_str"].apply(_assign_fin_math_group)
fin_math = fin_math[fin_math["field_group"].notna()]
fin_grouped = (
    fin_math.groupby(["awlevel_group", "year", "field_group"], as_index=False)[tabfield]
    .sum()
    .rename(columns={tabfield: "completions"})
)
baseline_fin = fin_grouped[fin_grouped["year"] == 2010][["awlevel_group", "field_group", "completions"]].rename(
    columns={"completions": "base_2010"}
)
fin_grouped = fin_grouped.merge(baseline_fin, on=["awlevel_group", "field_group"], how="left")
fin_grouped["rel_to_2010"] = fin_grouped["completions"] / fin_grouped["base_2010"]

fin_grouped = fin_grouped[fin_grouped['awlevel_group']=="Master"]

fig2, axes2 = plt.subplots(1, len(fin_grouped["awlevel_group"].unique()), figsize=(11, 6), sharey=True)
if not isinstance(axes2, np.ndarray):
    axes2 = np.array([axes2])
for ax, degree in zip(axes2, sorted(fin_grouped["awlevel_group"].unique())):
    subset = fin_grouped[fin_grouped["awlevel_group"] == degree]
    for group_name, grp in subset.groupby("field_group"):
        ax.plot(grp["year"], grp["completions"], marker="o", label=group_name)
    ax.axvline(2014, color = 'red', linestyle='--', linewidth=1)
    # add grey box between 2014 and 2016
    ax.axvspan(2014, 2016, color='gray', alpha=0.3)
    ax.axvline(2016, color = 'red', linestyle='--', linewidth=1)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Total Master's Degrees")
    ax.set_xlabel("Year")
    ax.legend()
fig2.tight_layout()
fig2.savefig(f"{OUTFIG_PATH}/ipeds_ma_fin_math_completions_by_year.png")
    

### random
con = ddb.connect()
FOIA_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
foia_raw = con.read_parquet(FOIA_PATH).df()

### among those with program end year = year, histogram of program length in months
foia_raw['program_length_months'] = (pd.to_datetime(foia_raw['program_end_date']) - pd.to_datetime(foia_raw['program_start_date'])).dt.days / 30.44
foia_programs = foia_raw[foia_raw['program_end_year']==foia_raw['year']].copy()
plt.figure()
g4_1 = sns.histplot(data=foia_programs, x='program_length_months', bins=50, kde=False)
