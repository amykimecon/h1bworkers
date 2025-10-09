import pandas as pd
import sys 
import os 
import re
import seaborn as sns 
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

# reading in raw data (read cipcode as string to preserve decimals)
allyrs_raw = [pd.read_excel(f"{root}/data/raw/ipeds/{filename}", sheet_name = 'TableLibrary', names = ['group', 'awardlvl', 'ciptitle', 'cipcode', 'total', 'native', 'asian', 'black', 'hispanic', 'pacificislander', 'white', 'twoormore', 'unknown', 'nonresident'], dtype = {'cipcode': str}, skiprows = 5).assign(year = int('20' + re.findall('ipeds_([0-9]{2}).xlsx', filename)[0])) for filename in os.listdir(f"{root}/data/raw/ipeds") if re.match('ipeds_([0-9]{2}).xlsx',filename) is not None]

# concatenating
ipeds = pd.concat(allyrs_raw)

# only keeping research/scholarship doctorate degrees
ipeds = ipeds[~((ipeds['awardlvl'] == 'Doctor\'s degree - professional practice') | (ipeds['awardlvl'] == 'Doctor\'s degree - other') | (ipeds['awardlvl'] == 'Doctor\'s degree - other (new degree classification)') | (ipeds['awardlvl'] == 'Doctor\'s degree - professional practice (new degree classification)'))]

# relabelling
ipeds['awardlvl'] = ipeds['awardlvl'].replace({
    'Doctor\'s degree - research/scholarship ': 'Doctor\'s degree', 
    'Doctor\'s degree - research/scholarship (new degree classification)': 'Doctor\'s degree'})

# indicator for whether cipcode is two or four-digit by checking if string has decimal point
ipeds['cip4'] = ipeds['cipcode'].astype(str).str.contains('\\.')

# combine bottom 2-digit cip codes into "Other"
bottom_cips = ipeds[~ipeds['cip4']].groupby('cipcode')['total'].sum().sort_values(ascending = True).head(25).index 
ipeds['othergrp'] = ipeds.apply(lambda row: 0 if row['cipcode'] not in bottom_cips else 1, axis = 1)   
ipeds['cipcode'] = ipeds.apply(lambda row: '0' if row['othergrp'] == 1 not in bottom_cips else row['cipcode'], axis = 1)   
ipeds['ciptitle'] = ipeds.apply(lambda row: 'Other' if row['othergrp'] == 1 else row['ciptitle'], axis = 1)  

# regrouping cip codes by cip title
ipeds = ipeds.groupby(['year', 'group', 'awardlvl', 'ciptitle', 'cipcode', 'cip4']).agg({'total': 'sum', 'native': 'sum', 'asian': 'sum', 'black': 'sum', 'hispanic': 'sum', 'pacificislander': 'sum', 'white': 'sum', 'twoormore': 'sum', 'unknown': 'sum', 'nonresident': 'sum'}).reset_index()

# converting each variable value to be relative to 2011 value 
ipeds = ipeds.merge(ipeds[ipeds['year'] == 2011][['group', 'awardlvl', 'ciptitle', 'cipcode', 'total', 'nonresident']].rename(columns = {'total': 'total_2011', 'nonresident': 'nonresident_2011'}), on = ['group', 'awardlvl', 'ciptitle', 'cipcode'], how = 'left')
ipeds['total_rel2011'] = ipeds['total'] / ipeds['total_2011']
ipeds['nonresident_rel2011'] = ipeds['nonresident'] / ipeds['nonresident_2011']
ipeds.drop(columns = ['total_2011', 'nonresident_2011'], inplace = True)

# variable for share international
ipeds['share_intl'] = ipeds['nonresident'] / ipeds['total']

# splitting into group
ipeds_princeton = ipeds[ipeds['group'] == 'ggregate Result - User-Selected Institutions']
ipeds_usc = ipeds[ipeds['group'] == 'University of Southern California']
ipeds = ipeds[ipeds['group'] == 'Aggregate Result - Title IV, degree-granting institutions']

# graphing total awards by year for two-digit cip codes by cip title and degree type, legend below graph
sns.lineplot(data = ipeds[(ipeds['cip4'] == False) & (ipeds['awardlvl'] == 'Master\'s degree')].groupby(['year', 'ciptitle', 'awardlvl'])['total_rel2011'].sum().reset_index(), x = 'year', y = 'total_rel2011', hue = 'ciptitle')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.scatterplot(data = ipeds[(ipeds['cip4'] == False) & (ipeds['awardlvl'] == 'Master\'s degree')].groupby(['year', 'ciptitle', 'awardlvl'])['total_rel2011'].sum().reset_index(), x = 'year', y = 'total_rel2011', hue = 'ciptitle')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

sns.scatterplot(data = ipeds[(ipeds['cip4'] == False) & (ipeds['awardlvl'] == 'Master\'s degree')].groupby(['year', 'ciptitle', 'awardlvl'])['total'].sum().reset_index(), x = 'year', y = 'total', hue = 'ciptitle')



ipeds[ipeds['cip4'] == False].groupby(['year', 'ciptitle', 'awardlvl'])['total'].sum().reset_index().pivot(index = 'year', columns = ['ciptitle', 'awardlvl'], values = 'total').plot() 
