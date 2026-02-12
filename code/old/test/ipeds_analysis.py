import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *

#raw_labs = pd.read_stata("/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/data/raw/ipeds/completions_all.dta", columns = ['cipcode', 'awlevel', 'cip2dig', 'year', 'progid']).rename(columns = {"cipcode": "cipcode_lab", "awlevel": "awlevel_lab", "cip2dig": "cip2dig_lab"})
raw_labs = pd.read_stata(f"{root}/data/raw/ipeds_completions_all.dta", columns = ['cipcode', 'awlevel', 'cip2dig', 'year', 'progid']).rename(columns = {"cipcode": "cipcode_lab", "awlevel": "awlevel_lab", "cip2dig": "cip2dig_lab"})

#raw = pd.read_stata("/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/data/raw/ipeds/completions_all.dta", convert_categoricals=False).merge(raw_labs, on = ["year", "progid"])
raw = pd.read_stata(f"{root}/data/raw/ipeds_completions_all.dta", convert_categoricals=False).merge(raw_labs, on = ["year", "progid"])

clean = raw[(raw['majornum']==1)&(raw['ctotalt'] > 0)&(raw['year']>=2002)]

ma_only = clean[(clean['master']==1)&(clean['cipcode']>=10000)]
ma_only['intlcat'] = np.where(pd.isnull(ma_only['share_intl'])==1, "null", np.where(ma_only['share_intl'] == 0, "No International Students", np.where(ma_only['share_intl'] >= 0.7, "70%+ International", np.where(ma_only['share_intl']>=0.3, "30-69% International", "1-29% International"))))

phd_only = clean[(clean['phd']==1)&(clean['cipcode']>=10000)]
ma_only_nodist = ma_only[ma_only['pmastrde'] == 0]

ma_only[['unitid','cipcode','cnralt','ctotalt','ptotal', 'pmastrde', 'share_intl', 'intlcat', 'cip2dig', 'year']].to_parquet(f"{root}/data/int/ipeds_ma_only.parquet", index = False)
# # subsetting to only programs that have at least 10 people in every year
# prog_counts = ma_only[(ma_only['year']>=2014)&(ma_only['year']<=2024)].groupby('progid').agg({'year': 'count', 'ctotalt': 'min'}).reset_index()
# prog_counts = prog_counts[(prog_counts['year'] == 11)&(prog_counts['ctotalt'] >= 5)]
# ma_only_panel = ma_only[ma_only['progid'].isin(prog_counts['progid'])]
# ma_only_nodist_panel = ma_only_nodist[ma_only_nodist['progid'].isin(prog_counts['progid'])]

# # Dodged histograms of share_intl in 2014, 2024
# sns.histplot(data = ma_only[(ma_only['year'].isin([2014, 2024]))&(ma_only['ctotalt'] >= 10)], x = 'share_intl', hue = 'year', multiple = "dodge", bins = 10, shrink = 0.9, stat = 'count', common_norm = False)
# plt.xlim(0,1)   
# plt.xlabel('Share of International Students')
# plt.ylabel('Number of Programs with 10+ Students')

# # same for non-dist programs
# sns.histplot(data = ma_only_nodist[(ma_only_nodist['year'].isin([2014, 2024]))&(ma_only_nodist['ctotalt'] >= 10)&(ma_only_nodist['share_intl']>0)], x = 'share_intl', hue = 'year', multiple = "dodge", bins = 10, shrink = 0.9, stat = 'count', common_norm = False)

# # add labels
# plt.xlim(0,1)
# plt.xlabel('Share of International Students')
# plt.ylabel("# In-Person Master's Programs with 10+ Students")
# plt.legend(title='Year of Graduation', labels=['2014', '2024'])
# # save plot to file
# plt.savefig('/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/figures/ipeds_shareintl_hist.png', bbox_inches='tight', dpi = 300)

# # among programs with 50%+ intl students in 2024, graph histogram of program size in 2024
# highintl_2024 = ma_only[(ma_only['year']==2024)&(ma_only['share_intl']>=0.5)&(ma_only['ctotalt']>=5)]
# sns.histplot(data = highintl_2024, x = 'ctotalt', bins = range(0, 105, 5), discrete = True)
# plt.xlabel('Total Number of Students in 2024')
# plt.ylabel('Number of Programs with 50%+ International Students')

# # among programs with 50%+ intl students in 2024, graph histogram of first year with 5+ students
# highintl_2024 = ma_only[(ma_only['year']==2024)&(ma_only['share_intl']>=0.5)&(ma_only['ctotalt']>=5)]
# first_years = ma_only[(ma_only['ctotalt']>=10)&(ma_only['year']>=2004)&(ma_only['share_intl']>=0.5)&(ma_only['progid'].isin(highintl_2024['progid']))].groupby('progid').agg({'year': 'min', 'ctotalt':'mean'}).reset_index()
# sns.histplot(data = first_years, x = 'year', bins = range(2002, 2025), discrete = True)
# plt.xlabel('First Year as IHMA with 5+ Students')
# plt.ylabel('Number of Programs')

# # among programs with 50%+ intl students in 2024, graph histogram of share intl in 2014
# highintl_2024 = ma_only[(ma_only['year']==2024)&(ma_only['share_intl']>=0.5)&(ma_only['ctotalt']>=5)]
# intl_2014 = ma_only[(ma_only['year']==2014)&(ma_only['progid'].isin(highintl_2024['progid']))]
# sns.histplot(data = intl_2014, x = 'share_intl', bins = 10)
# plt.xlim(0,1)
# plt.xlabel('Share of International Students in 2014')
# plt.ylabel('Number of Programs')

# # graphing line plot of share of intl-heavy programs and share of intl students over time
# df = ma_only
# df_nodist = ma_only_nodist

# sns.lineplot(data = df.assign(highintl = df['share_intl'] >= 0.5).groupby('year').agg({'highintl': 'sum', 'progid': 'count'}).assign(share_high_intl = lambda x: x['highintl']/x['progid']), x = 'year', y = 'share_high_intl', label = 'Share of IHMA Programs')

# sns.lineplot(data = df_nodist.assign(highintl = df_nodist['share_intl'] >= 0.5).groupby('year').agg({'highintl': 'sum', 'progid': 'count'}).assign(share_high_intl = lambda x: x['highintl']/x['progid']), x = 'year', y = 'share_high_intl', label = 'Share of IHMA Programs (In-Person Only)')
# # adding line for share of intl students overall
# sns.lineplot(data = df.groupby('year').agg({'ctotalt': 'sum', 'cnralt': 'sum'}).reset_index().assign(share_intl = lambda x: x['cnralt'] / x['ctotalt']), x = 'year', y = 'share_intl', linestyle = '--', label = 'Share of MA Intl Students')

# sns.lineplot(data = df_nodist.groupby('year').agg({'ctotalt': 'sum', 'cnralt': 'sum'}).reset_index().assign(share_intl = lambda x: x['cnralt'] / x['ctotalt']), x = 'year', y = 'share_intl', linestyle = '--', label = 'Share of MA Intl Students (In-Person Only)')
# # save plot to file
# plt.ylabel("Share")
# plt.xlabel("Year of Graduation")
# plt.savefig('/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/figures/ipeds_shareintl_ihma_over_time.png', bbox_inches='tight', dpi = 300)

# ## BOTH PLOTS SIDE BY SIDE
# # setting up the figure and axes
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # --- Left plot: MA ---
# df = ma_only
# grp_cip = df[(df['cip2dig_lab'].isin([
#     "Computer Science", "Engineering", "Math", "Physical Sciences",
#     "Social Sciences", "Healthcare", "Business"
# ]))].groupby(['year','cip2dig_lab'], observed=True) \
#   .agg({'ctotalt': 'sum', 'cnralt': 'sum'}).reset_index() \
#   .assign(share_intl=lambda x: x['cnralt'] / x['ctotalt'])

# grp_cip['cip2dig_lab'] = pd.Categorical(
#     grp_cip['cip2dig_lab'],
#     categories=["Computer Science", "Engineering", "Math",
#                 "Physical Sciences", "Social Sciences",
#                 "Healthcare", "Business"],
#     ordered=True
# )

# ax1 = sns.lineplot(data=grp_cip, x='year', y='share_intl', hue='cip2dig_lab', ax=axes[0])
# sns.lineplot(
#     data=df.groupby('year').agg({'ctotalt':'sum','cnralt':'sum'}).reset_index()
#        .assign(share_intl=lambda x: x['cnralt']/x['ctotalt']),
#     x='year', y='share_intl', color='black', linestyle='--', label='Overall', ax=ax1
# )
# ax1.set_xlabel("Year of Graduation")
# ax1.set_ylabel("Share of International Master's Students")
# ax1.legend_.remove()

# # --- Right plot: PhDs ---
# df = phd_only
# grp_cip = df[(df['cip2dig_lab'].isin([
#     "Computer Science", "Engineering", "Math", "Physical Sciences",
#     "Social Sciences", "Healthcare", "Business"
# ]))].groupby(['year','cip2dig_lab'], observed=True) \
#   .agg({'ctotalt': 'sum', 'cnralt': 'sum'}).reset_index() \
#   .assign(share_intl=lambda x: x['cnralt'] / x['ctotalt'])

# grp_cip['cip2dig_lab'] = pd.Categorical(
#     grp_cip['cip2dig_lab'],
#     categories=["Computer Science", "Engineering", "Math",
#                 "Physical Sciences", "Social Sciences",
#                 "Healthcare", "Business"],
#     ordered=True
# )

# ax2 = sns.lineplot(data=grp_cip, x='year', y='share_intl', hue='cip2dig_lab', ax=axes[1])
# sns.lineplot(
#     data=df.groupby('year').agg({'ctotalt':'sum','cnralt':'sum'}).reset_index()
#        .assign(share_intl=lambda x: x['cnralt']/x['ctotalt']),
#     x='year', y='share_intl', color='black', linestyle='--', label='Overall', ax=ax2
# )
# ax2.set_xlabel("Year of Graduation")
# ax2.set_ylabel("Share of International PhD Students")
# ax2.legend_.remove()

# # --- Shared legend ---
# handles, labels = ax2.get_legend_handles_labels()
# fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc="center left")

# plt.tight_layout()

# # save plot to file
# plt.savefig('/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/figures/ipeds_shareintl_ma_phd_over_time.png', bbox_inches='tight', dpi = 300)
# plt.show()


# # share of intl students over time, grouped by program subject (four-digit cip code), excluding categories with ctotalt == 0
# df = ma_only
# grp_cip = df[(df['cip4dig'].isin([1107, 1401, 1409, 1419, 1437,2701,2703,4506,5208,5213]))].groupby(['year','cip4dig'], observed = True).agg({'ctotalt': 'sum', 'cnralt': 'sum'}).reset_index().assign(share_intl = lambda x: x['cnralt'] / x['ctotalt'])
# grp_cip['cip4dig_lab'] = grp_cip['cip4dig'].replace({
#     1107: 'Computer Science',
#     1401: 'General Engineering',
#     1409: 'Electrical Eng',
#     1419: 'Mechanical Eng',
#     1437: 'ORFE',
#     2701: 'General Math',
#     2703: 'Applied Math',
#     4506: 'Economics',
#     5208: 'Finance',
#     5213: 'Management Science'
# })
# ax = sns.lineplot(data = grp_cip, x = 'year', y = 'share_intl', hue = 'cip4dig_lab')

# sns.lineplot(data = df.groupby('year').agg({'ctotalt': 'sum', 'cnralt': 'sum'}).reset_index().assign(share_intl = lambda x: x['cnralt'] / x['ctotalt']), x = 'year', y = 'share_intl', color = 'black', linestyle = '--', label = 'Overall', ax = ax)

# ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
# plt.xlabel("Year of Graduation")
# plt.ylabel("Share of International MA Students")
# plt.savefig('/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/figures/ipeds_shareintl_cip4dig_over_time.png', bbox_inches='tight', dpi = 300)
# plt.show()


# # share of all intl students in ma programs, phd, ba, ihma
# df = clean
# df['awlevel_group'] = np.where(df['awlevel'].isin([5, 7, 17, 9]), df['awlevel'].replace({
#     5: 'Bachelor',
#     7: 'Master',
#     17: 'Doctor',
#     9: 'Doctor'
# }), 'Other')
# df['cnralt_ihma'] = np.where((df['share_intl'] >= 0.5)&(df['awlevel_group']=='Master'), df['cnralt'], 0)
# grp_cip = df.groupby(['year','awlevel_group'], observed = True).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'cnralt_ihma':'sum'}).reset_index()

# grp_cip['cnralt_tot'] = grp_cip.groupby('year')['cnralt'].transform('sum')
# grp_cip['share_of_intl'] = grp_cip['cnralt']/grp_cip['cnralt_tot']
# grp_cip['share_ihma_of_intl'] = grp_cip['cnralt_ihma']/grp_cip['cnralt_tot']
# ax = sns.lineplot(data = grp_cip, x = 'year', y = 'share_of_intl', hue = 'awlevel_group')

# sns.lineplot(data = grp_cip[grp_cip['awlevel_group']=="Master"], x = 'year', y = 'share_ihma_of_intl', color = 'black', linestyle = '--', label = 'Intl-Heavy Master', ax = ax)

# ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
# plt.xlabel("Year of Graduation")
# plt.ylabel("Share of International Students by Degree Type")
# plt.savefig('/Users/amykim/Princeton Dropbox/Amy Kim/intl_students/figures/ipeds_share_deg_intl.png', bbox_inches='tight', dpi = 300)
# plt.show()