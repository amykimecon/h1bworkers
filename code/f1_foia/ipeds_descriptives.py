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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 
INT_FOLDER = f"{root}/data/int/int_files_nov2025"

# READING IN RAW DATA
ipeds = pd.read_parquet(f"{INT_FOLDER}/ipeds_completions_all.parquet")
ipeds['ihmp'] = (ipeds['share_intl'] >= 0.5)&(ipeds['awlevel'] == 7)&(ipeds['majornum'] == 1)&(ipeds['STEMOPT'] == 1)
ipeds['awlevel_group'] = np.where(ipeds['awlevel'].isin([5, 7, 17, 9]), ipeds['awlevel'].replace({
    5: 'Bachelor',
    7: 'Master',
    17: 'Doctor',
    9: 'Doctor'
}), 'Other')
ipeds['intl_ihmp'] = ipeds['ihmp']*ipeds['cnralt']

# DESCRIPTIVE ONE: Share of all degree completions by international students over time
ipeds_by_year = ipeds.groupby(['year']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_year['share_intl'] = ipeds_by_year['cnralt'] / ipeds_by_year['ctotalt']
ipeds_by_year['share_ihmp_intl'] = ipeds_by_year['intl_ihmp'] / ipeds_by_year['ctotalt']
ipeds_by_deg_year = ipeds.groupby(['year', 'awlevel_group']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_deg_year['share_intl'] = ipeds_by_deg_year['cnralt'] / ipeds_by_deg_year['ctotalt']
sns.lineplot(data=ipeds_by_year, x='year', y='share_intl', color = 'gray')
sns.lineplot(data=ipeds_by_deg_year, x='year', y='share_intl', hue='awlevel_group')

sns.lineplot(data=ipeds_by_year, x='year', y='cnralt', color = 'gray')
sns.lineplot(data=ipeds_by_deg_year, x='year', y='cnralt', hue='awlevel_group')

# DESCRIPTIVE TWO: Share of Master's IHMPs over time
ipeds_masters = ipeds[ipeds['awlevel'] == 7]
ipeds_masters_by_year = ipeds_masters.groupby(['year']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_masters_by_year['share_intl'] = ipeds_masters_by_year['cnralt'] / ipeds_masters_by_year['ctotalt