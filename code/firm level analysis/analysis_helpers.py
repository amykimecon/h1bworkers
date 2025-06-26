import duckdb as ddb
import pandas as pd
import numpy as np
import fiscalyear
import seaborn as sns

## HELPERS
# get fiscal year from date (pandas)
def get_fiscal_year(date):
    if pd.isnull(date):
        return None
    return fiscalyear.FiscalDate(date.year, date.month, date.day).fiscal_year

# get fiscal year from date (sql)
def get_fiscal_year_sql(date):
    month = int(date[5:7])
    year = int(date[:4])
    if month >= 10:
        return year + 1
    else:
        return year 
    
# get fiscal year from foia date (sql)
def get_fiscal_year_foia_sql(date):
    splits = date.split("/")
    month = int(splits[0])
    year = int(splits[2])
    if month >= 10:
        return year + 1
    else:
        return year 
    
# get quarter from date (sql)
def get_quarter_sql(date):
    month = int(date[5:7])
    year = int(date[:4])
    return year + np.floor((month-1)/3)/4

# get quarter from foia date (sql)
def get_quarter_foia_sql(date):
    splits = date.split("/")
    month = int(splits[0])
    year = int(splits[2])
    return year + np.floor((month-1)/3)/4

def get_date_foia_sql(date):
    splits = date.split("/")
    month = int(splits[0])
    year = int(splits[2])
    day = int(splits[1])
    return f"{year}-{month:02}-{day:02}"

# get fiscal year from quarter (sql)
def get_fy_from_quarter_sql(q):
    return(np.floor(q+0.25))

# get sql strings for getting month, day, year from FOIA Bloomberg format dates
def get_info_from_bb_date(datecol, unit):
    if unit == 'month':
        return(f"CASE WHEN {datecol} = 'NA' THEN NULL ELSE SUBSTRING({datecol}, 1, STRPOS({datecol}, '/') -1)::INT END")    
    elif unit == 'day':
        return(f"CASE WHEN {datecol} = 'NA' THEN NULL ELSE SUBSTRING({datecol}, STRPOS({datecol}, '/') + 1, STRPOS(SUBSTRING({datecol}, STRPOS({datecol}, '/') + 1, 3), '/') - 1)::INT END")
    elif unit == 'year':
        return(f"CASE WHEN {datecol} = 'NA' THEN NULL ELSE SUBSTRING({datecol}, STRPOS({datecol}, '/') + STRPOS(SUBSTRING({datecol}, STRPOS({datecol}, '/') + 1, 3), '/') + 1, 4)::INT END")
    print("ERROR WITH UNIT TYPE!")

## MORE HELPERS
def bin_long_diffs(df, varlist):
    pivot_long = df.pivot(index = 't', columns = ['treat'], values = varlist)
    pivot_long = pivot_long.join(pivot_long.groupby(level=0,axis=1).diff().rename(columns={'1':'diff'}).loc(axis = 1)[:,'diff']).reset_index().melt(id_vars = [('t','')])

    pivot_long.columns = ['t', 'var', 'treat', 'value']

    rawvars = pivot_long.loc[pivot_long['treat'] != 'diff']
    diffs = pivot_long.loc[pivot_long['treat'] == 'diff']

    return (rawvars, diffs)

def graph_df_nohue(df, ly):
    g = sns.FacetGrid(data = df, row = 'var', sharey = False, height = 2, aspect = 2)
    g.map(sns.scatterplot, 't', 'value')
    g.refline(x = ly - 0.75)
    return g

def graph_df_hue(df, ly, hue):
    g = sns.FacetGrid(data = df, hue = hue, row = 'var', sharey = False, height = 2, aspect = 2)
    g.map(sns.scatterplot, 't', 'value').add_legend()
    g.refline(x = ly - 0.75)
    return g