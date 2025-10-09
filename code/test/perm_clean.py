import pandas as pd
import sys 
import os 
import re
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.formula.api as smf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

fromscratch = False 

#############################################
## READING AND CONCATENATING RAW PERM DATA ##
#############################################
# column name harmonization across years of PERM data
colmap = {
    "CASE_NO": "CASE_NUMBER",
    "CASE_STATUS": "CASE_STATUS",
    "CASE_RECEIVED_DATE": "RECEIVED_DATE",
    "APPLICATION_TYPE": "APPLICATION_TYPE",
    "DECISION_DATE": "DECISION_DATE",
    "REFILE": "REFILE",
    "ORIG_FILE_DATE": "ORIG_FILE_DATE",
    "EMPLOYER_NAME": "EMPLOYER_NAME",
    "PW_TRACK_NUM": "PW_TRACK_NUMBER",
    "PW_AMOUNT_9089": "PW_WAGE",
    "PW_UNIT_OF_PAY_9089": "PW_UNIT_OF_PAY",
    "PW_DETERM_DATE": "PW_DETERMINATION_DATE",
    "PW_EXPIRE_DATE": "PW_EXPIRATION_DATE",
    "WAGE_OFFER_FROM_9089": "WAGE_OFFER_FROM",
    "WAGE_OFFER_TO_9089": "WAGE_OFFER_TO",
    "WAGE_OFFER_UNIT_OF_PAY_9089": "WAGE_OFFER_UNIT_OF_PAY",
    "JOB_INFO_JOB_TITLE": "JOB_TITLE",
    "JOB_INFO_EDUCATION": "MINIMUM_EDUCATION",
    "JOB_INFO_MAJOR": "MAJOR_FIELD_OF_STUDY",
    "JOB_INFO_EXPERIENCE": "REQUIRED_EXPERIENCE",
    "JOB_INFO_EXPERIENCE_NUM_MONTHS": "REQUIRED_EXPERIENCE_MONTHS",
    "SPECIFIC_SKILLS":"SPECIFIC_SKILLS",
    "RECR_INFO_PROFESSIONAL_OCC": "PROFESSIONAL_OCCUPATION",
    "RECR_INFO_COLL_UNIV_TEACHER": "APP_FOR_COLLEGE_U_TEACHER",
    "RI_COLL_TEACH_SELECT_DATE": "TEACHER_SELECT_DATE",
    "RECR_INFO_SWA_JOB_ORDER_START": "SWA_JOB_ORDER_START_DATE",
    "RECR_INFO_SWA_JOB_ORDER_END": "SWA_JOB_ORDER_END_DATE",
    "RECR_INFO_FIRST_AD_START": "FIRST_ADVERTISEMENT_START_DATE",
    "RI_JOB_SEARCH_WEBSITE_FROM": "JOB_SEARCH_WEBSITE_FROM_DATE",
    "RI_JOB_SEARCH_WEBSITE_TO": "JOB_SEARCH_WEBSITE_TO_DATE",
    "RI_LAYOFF_IN_PAST_SIX_MONTH": "LAYOFF_IN_PAST_SIX_MONTHS",
    "COUNTRY_OF_CITIZENSHIP": "COUNTRY_OF_CITIZENSHIP",
    "COUNTRY_OF_CITZENSHIP": "COUNTRY_OF_CITIZENSHIP",
    "CLASS_OF_ADMISSION": "CLASS_OF_ADMISSION",
    "FOREIGN_WORKER_INFO_EDUCATION": "FOREIGN_WORKER_EDUCATION",
    "FOREIGN_WORKER_INFO_MAJOR": "FOREIGN_WORKER_INFO_MAJOR",
    "FW_INFO_YR_REL_EDU_COMPLETED": "FOREIGN_WORKER_YRS_ED_COMP",
    "FOREIGN_WORKER_INFO_INST": "FOREIGN_WORKER_INST_OF_ED",
    "FOREIGN_WORKER_ED_INST_COUNTRY": "FOREIGN_WORKER_ED_INST_COUNTRY",
    "FOREIGN_WORKER_CURR_EMPLOYED":"FOREIGN_WORKER_CURR_EMPLOYED"
}

if fromscratch:
    # reading in raw data
    allyrs_raw = [pd.read_excel(f"{root}/data/raw/perm/{filename}").rename(columns = lambda x: x.strip().upper().replace(" ", "_")).rename(columns=colmap).assign(year = int('20' + re.search('FY([0-9]{2})?([0-9]{2})', filename).group(2)))[lambda d: d.columns.intersection(colmap.values()).union(['year'])] for filename in os.listdir(f"{root}/data/raw/perm") if re.match('PERM_.*.xlsx',filename) is not None]

    # concatenating
    perm = pd.concat(allyrs_raw).drop_duplicates()

    # write perm to parquet
    perm.apply(lambda col: col.where(col.notnull(), None).astype(str) if col.dtype == 'object' else col).to_parquet(f'{root}/data/int/perm_agg_raw.parquet')
else:
    perm = pd.read_parquet(f'{root}/data/int/perm_agg_raw.parquet')

########################
## CLEANING PERM DATA ##
########################
# cleaning (TODO: take earliest decision per case number)
perm['processing_time'] = (perm['DECISION_DATE'] - perm['RECEIVED_DATE']).dt.days

perm['rec_month'] = perm['RECEIVED_DATE'].dt.month
perm['dec_month'] = perm['DECISION_DATE'].dt.month
perm['rec_yr'] = perm['RECEIVED_DATE'].dt.year
perm['dec_yr'] = perm['DECISION_DATE'].dt.year
perm['denied'] = perm['CASE_STATUS'] == 'Denied'
perm['withdrawn'] = perm['CASE_STATUS'] == 'Withdrawn'

perm['h1b'] = perm['CLASS_OF_ADMISSION'] == 'H-1B'
perm['india'] = perm['COUNTRY_OF_CITIZENSHIP'] == 'INDIA'

# iding top firms, firms of interest
perm['fb'] = perm['EMPLOYER_NAME'].str.lower().str.contains('facebook|meta', regex = True)
perm['apple'] = perm['EMPLOYER_NAME'].str.lower().str.contains('apple')
perm['bigtech_other'] = perm['EMPLOYER_NAME'].str.lower().str.contains('google|amazon|microsoft|intel')

topcompanies = perm.groupby('EMPLOYER_NAME').size().nlargest(25).index 
perm['topcompany'] = perm['EMPLOYER_NAME'].isin(topcompanies)

perm_full = perm.copy()
perm = perm[perm['RECEIVED_DATE']>='2015-01-01']

##########################################
## HELPER FUNCTIONS: GROUPING, PLOTTING ##
##########################################
# helper function to get list of years where variable var is nonmissing
def yrs_nonmissing(var, df = perm):
    print(df[(pd.isnull(df[var])==0)&(df[var]!='None')]['year'].sort_values().unique())

# helper function to group by time interval
def grp_perm(grpvar, freq, othergrpvars = [], mindate = '2015-01-01', maxdate = '2025-01-01', df = perm, agg_dict = dict()):
    agg_dict['n'] = ('CASE_NUMBER', 'nunique')
    agg_dict["avg_processing_time"] = ('processing_time', 'mean')
    agg_dict["med_processing_time"] = ('processing_time', 'median')
    agg_dict['denial_rate'] = ('denied','mean')
    agg_dict['share_topcompany'] = ('topcompany', 'mean')

    if grpvar == 'RECEIVED_DATE':
        agg_dict['avg_decision_date'] = ('DECISION_DATE', 'mean')
        agg_dict['med_decision_date'] = ('DECISION_DATE', 'median')
    elif grpvar == 'DECISION_DATE':
        agg_dict['avg_received_date'] = ('RECEIVED_DATE', 'mean')
        agg_dict['med_received_date'] = ('RECEIVED_DATE', 'median')

    grped = df.groupby([pd.Grouper(key = grpvar, freq = freq)] + othergrpvars).agg(**{
        name: pd.NamedAgg(column = col, aggfunc = func) for name, (col, func) in agg_dict.items()
    }
    ).reset_index()

    return grped[grped[grpvar].between(mindate, maxdate)]

# helper function to plot (with option to plot on two axes)
def plot_time(df, xvar, yvar1, ylab1 = None, yvar2 = None, ylab2 = None, hue = None, noweekends = True, rollyvar1 = None, vertlines = 'jun30', xlab = None, rollyvar2 = None, figsave = None, dir = '/Users/amykim/Documents/GitHub/h1bworkers/output/slides/slides_25sep25'):
    if noweekends == True:
        df = df[df[xvar].dt.weekday < 5]
    fig, ax1 = plt.subplots(figsize = (10,5))
    if ylab1 is None:
        ylab1 = yvar1 

    if rollyvar1 is not None:
        df[yvar1] = df[yvar1].rolling(rollyvar1).mean()

    if rollyvar2 is not None:
        df[yvar2] = df[yvar2].rolling(rollyvar2).mean()

    if hue is None:
        sns.scatterplot(df, x = xvar, y = yvar1, color = 'blue', ax = ax1)
    else:
        sns.scatterplot(df, x = xvar, y = yvar1, hue = hue, ax = ax1)
    plt.xticks(rotation = 45)
    ax1.set_ylabel(ylab1)

    if yvar2 is not None:
        ax1.set_ylabel(ylab1, color = 'blue')
        ax1.tick_params(axis = 'y',labelcolor='blue')

        if ylab2 is None:
            ylab2 = yvar2 

        ax2 = ax1.twinx()
        if hue is None:
            sns.scatterplot(df, x = xvar, y = yvar2, color = 'orange', ax = ax2)
        else:
            sns.scatterplot(df, x = xvar, y = yvar2, hue = hue, ax = ax2)
        ax2.set_ylabel(ylab2, color = 'orange')
        ax2.tick_params(axis = 'y',labelcolor='orange')
    
    if vertlines == 'monthstart':
        first_of_month = pd.date_range(df[xvar].min(), df[xvar].max(), freq='MS')  # MS = Month Start
        for date in first_of_month:
            ax1.axvline(date, color='grey', linestyle='--', alpha=0.7)
    elif vertlines == 'jun30':
        yrs = range(df[xvar].dt.year.min(), df[xvar].dt.year.max()+1)
        june30s = [pd.Timestamp(f"{year}-06-30") for year in yrs]
        for d in june30s:
            ax1.axvline(x = d, color = 'grey', linestyle = '--', alpha = 0.7)

    if xlab is not None:
        ax1.set_xlabel(xlab)

    if figsave is not None:
        plt.savefig(f'{dir}/{figsave}.png', bbox_inches = 'tight')

    plt.show()

def rd_plot(yvar, x0, xvar = 'RECEIVED_DATE', days_bw = 30, df = perm, hue = None, xbins = 20, rawplot = False, grpby = '1D', xlab = None, ylab = None,figsave = None, dir = '/Users/amykim/Documents/GitHub/h1bworkers/output/slides/slides_25sep25'):
    mindate = pd.Timestamp(x0) - pd.Timedelta(days = days_bw)
    maxdate = pd.Timestamp(x0) + pd.Timedelta(days = days_bw)

    # if plotting raw values or total counts, group by day
    if yvar == 'n' or rawplot:
        if yvar in df.columns:
            agg_dict = {yvar: (yvar, 'mean')}
        else:
            agg_dict = dict()
        df_filt = grp_perm(xvar, grpby, mindate = mindate, maxdate = maxdate, othergrpvars=[hue] if hue is not None else [], agg_dict = agg_dict, df = df)
    else:
        df_filt = df[df[xvar].between(mindate, maxdate)]

    # if yvar is datetime, convert to ordinal
    if pd.api.types.is_datetime64_any_dtype(df_filt[yvar]):
        print('here')
        df_filt[yvar] = df_filt[yvar].map(pd.Timestamp.toordinal)

    if pd.api.types.is_bool_dtype(df_filt[yvar]):
        df_filt[yvar] = df_filt[yvar].astype(int)

    # filtering out outliers 
    lower = df_filt[yvar].quantile(0.01)
    upper = df_filt[yvar].quantile(0.99)
    df_filt = df_filt[(df_filt[yvar]>= lower) & (df_filt[yvar] <= upper)]

    df_filt['x_num'] = df_filt[xvar].map(pd.Timestamp.toordinal)
    df_filt['post'] = df_filt[xvar] >= x0
    fig, ax = plt.subplots(figsize = (10,5))
    
    if rawplot: 
        if hue is None:
            sns.scatterplot(data = df_filt, x = xvar, y = yvar, ax = ax, hue = 'post')
        else:
            sns.scatterplot(data = df_filt, x = xvar, y = yvar, ax = ax, hue = hue)
        
        ax.axvline(x = pd.Timestamp(x0), color = 'grey', linestyle = '--', alpha = 0.7)
        
    else:
        for lab, grp in df_filt.groupby('post'):
            if hue is None:
                sns.regplot(data = grp, x = 'x_num', y = yvar, ax = ax, x_bins = xbins, label = f"Post: {lab}")
            else:
                for lab2, grp2 in grp.groupby(hue):
                    sns.regplot(data = grp2, x = 'x_num', y = yvar, ax = ax, x_bins = xbins, label = f"Post: {lab}, {hue}: {lab2}")
        
        ax.axvline(x = pd.Timestamp(x0).toordinal(), color = 'grey', linestyle = '--', alpha = 0.7)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

        # convert numeric ticks back to dates
        tick_labels = [(pd.Timestamp.fromordinal(int(t))-pd.Timestamp(x0)).days for t in ax.get_xticks()]
        ax.set_xticklabels(tick_labels)

        ax.legend(title='Group')

    if xlab is not None:
        ax.set_xlabel(xlab)
    
    if ylab is not None:
        ax.set_ylabel(ylab)
    
    if figsave is not None:
        plt.savefig(f'{dir}/{figsave}.png', bbox_inches = 'tight')

    plt.show()

##########################################
## RESULTS FOR SLIDES
##########################################
# raw plot 1: processing times by receipt date
plot_time(grp_perm('RECEIVED_DATE','1W'), xvar = 'RECEIVED_DATE', yvar1 = 'med_processing_time', vertlines = False, xlab = 'Date Application Received by DOL', ylab1 = 'Median Processing Time (Days)', noweekends = False, figsave = 'perm_waittime_by_receiveddate_raw')

# raw plot 2: number of applications by receipt date 
plot_time(grp_perm('RECEIVED_DATE','1W'), xvar = 'RECEIVED_DATE', xlab = 'Date Application Received by DOL',  yvar1 = 'n', ylab1 = 'Number of Applications Received (10-week MA)', noweekends = False, figsave = 'perm_n_by_receiveddate_raw', rollyvar2 = 10)

# superimposing number of applications
plot_time(grp_perm('RECEIVED_DATE','1W'), xvar = 'RECEIVED_DATE', xlab = 'Date Application Received by DOL',  yvar1 = 'med_processing_time', ylab1 = 'Median Processing Time (Days)', yvar2 = 'n', ylab2 = 'Number of Applications Received', noweekends = False, figsave = 'perm_waittime_n_by_receiveddate_raw')

# reverse
plot_time(grp_perm('RECEIVED_DATE','1W'), xvar = 'RECEIVED_DATE', xlab = 'Date Application Received by DOL',  yvar2 = 'med_processing_time', ylab2 = 'Median Processing Time (Days)', yvar1 = 'n', ylab1 = 'Number of Applications Received (10-week MA)', noweekends = False, figsave = 'perm_n_waittime_by_receiveddate_raw', rollyvar1 = 10)


# zooming in on one of these dicontinuities: 
bw = 200
x0 = '2017-09-15'

## raw plot: processing time
rd_plot(yvar = 'processing_time', x0 = x0, days_bw = bw, rawplot = True, xlab = 'Date Application Received by DOL', ylab = 'Processing Time (Days)', figsave = 'sep17_waittime_raw')

## rd plot
rd_plot(yvar = 'processing_time', x0 = x0, days_bw = bw, xlab = f'Day Application Received by DOL Relative to {x0}', ylab = 'Processing Time (Days)', figsave= 'sep17_waittime_rd')

## rd plot: n
rd_plot(yvar = 'n', x0 = x0, days_bw = bw, xlab = f'Day Application Received by DOL Relative to {x0}', ylab = 'Number of Applications Received', figsave= 'sep17_n_rd')

## rd plot: denied
rd_plot(yvar = 'denied', x0 = x0, days_bw = bw, xlab = f'Day Application Received by DOL Relative to {x0}', ylab = 'Application Denial Rate', figsave= 'sep17_denied_rd')

## rd plot: topcompany
rd_plot(yvar = 'topcompany', x0 = x0, days_bw = bw, xlab = f'Day Application Received by DOL Relative to {x0}', ylab = 'Share of Applications from Top 25 Companies', figsave= 'sep17_topcompany_rd')

## facebook
plot_time(grp_perm('RECEIVED_DATE','1W', df = perm[perm['fb']==1], mindate = '2018-06-01'), xvar = 'RECEIVED_DATE', yvar1 = 'med_processing_time', vertlines = False, xlab = 'Date Application Received by DOL', ylab1 = 'Median Processing Time (Days)', noweekends = False, figsave = 'fb_waittime_by_receiveddate_raw')

plot_time(grp_perm('RECEIVED_DATE','1W', df = perm[perm['fb']==1], mindate = '2018-06-01'), xvar = 'RECEIVED_DATE', xlab = 'Date Application Received by DOL',  yvar2 = 'med_processing_time', ylab2 = 'Median Processing Time (Days)', yvar1 = 'n', ylab1 = 'Number of Applications Received (10-week MA)', noweekends = False, figsave = 'fb_n_waittime_by_receiveddate_raw', rollyvar1 = 10)


plot_time(grp_perm('RECEIVED_DATE','1W', df = perm[perm['fb']==1], mindate = '2018-06-01'), xvar = 'RECEIVED_DATE', yvar1 = 'med_processing_time', vertlines = False, xlab = 'Date Application Received by DOL', ylab1 = 'Median Processing Time (Days)', noweekends = False, figsave = 'fb_waittime_by_receiveddate_raw')

## fb rd: waittime
rd_plot(yvar = 'processing_time', x0 = '2020-12-01', days_bw = bw, rawplot = True, xlab = 'Date Application Received by DOL', ylab = 'Processing Time (Days)', figsave = 'fb_waittime_raw', df = perm[perm['fb']==1], xvar ='DECISION_DATE')

plot_time(grp_perm('RECEIVED_DATE','1W', df = perm[perm['fb']==1], mindate = '2019-06-01'), xvar = 'RECEIVED_DATE', yvar1 = 'n', vertlines = False, xlab = 'Date Application Received by DOL', ylab1 = 'Number of Applications', noweekends = False)


## 
yvar = 'processing_time'
mindate = '2017-01-15'
maxdate='2019-08-15'
if yvar == 'n':
    x = grp_perm('RECEIVED_DATE','1D', mindate = mindate,maxdate=maxdate)
    y = x[x['n']>100]
else:
    y = perm[(perm['RECEIVED_DATE'].dt.weekday < 5)&(perm['RECEIVED_DATE'].between(mindate, maxdate))]
y['month'] = y['RECEIVED_DATE'].dt.to_period("M")
y['x_num'] = y['RECEIVED_DATE'].map(pd.Timestamp.toordinal)
fig, ax = plt.subplots()
#sns.scatterplot(y, x = 'x_num', y = yvar, ax = ax)
for month, group in y.groupby('month'):
    sns.regplot(data =group, x = 'x_num', y = yvar, ax = ax, x_bins = 20)
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

# convert numeric ticks back to dates
tick_labels = pd.to_datetime([pd.Timestamp.fromordinal(int(t)) for t in ax.get_xticks()])
ax.set_xticklabels(tick_labels.strftime('%Y-%m-%d'), rotation=45)



# question 1: is there a discontinuity in the processing time or denial rate by submission/decision date?
#grpvar = 'DECISION_DATE'
grpvar = 'RECEIVED_DATE'

grp_day = perm.groupby(grpvar).agg(n = pd.NamedAgg('CASE_NUMBER', 'count'), avg_processing_time = pd.NamedAgg('processing_time', 'mean'),  med_processing_time = pd.NamedAgg('processing_time', 'median'), denial_rate = pd.NamedAgg('denied', 'mean')).reset_index().sort_values(grpvar)
grp_day['roll_n'] = grp_day['n'].rolling(3).mean()

sns.scatterplot(grp_day[(grp_day[grpvar].between("2021-09-01",'2022-03-01'))], x = grpvar, y = 'roll_n')
plt.xticks(rotation = 45)

sns.scatterplot(grp_day[(grp_day[grpvar].dt.year >= 2016)&(grp_day[grpvar].dt.year <= 2018)], x = grpvar, y = 'avg_processing_time')
plt.xticks(rotation = 45)

sns.scatterplot(grp_day[(grp_day[grpvar].dt.year >= 2016)&(grp_day[grpvar].dt.year <= 2018)], x = grpvar, y = 'denial_rate')
plt.xticks(rotation = 45)

grp_month_rec = perm.groupby([pd.Grouper(key = "RECEIVED_DATE", freq = '1M')]).agg(n = pd.NamedAgg('CASE_NUMBER', 'count'), avg_processing_time = pd.NamedAgg('processing_time', 'mean'), med_processing_time = pd.NamedAgg('processing_time', 'median'), denial_rate = pd.NamedAgg('denied', 'mean')).reset_index() 
grp_month_dec = perm.groupby([pd.Grouper(key = "DECISION_DATE", freq = '1M')]).agg(n = pd.NamedAgg('CASE_NUMBER', 'count'), avg_processing_time = pd.NamedAgg('processing_time', 'mean'), med_processing_time = pd.NamedAgg('processing_time', 'median'), denial_rate = pd.NamedAgg('denied', 'mean')).reset_index() 
grp_month = pd.merge(grp_month_rec, grp_month_dec, 'inner', left_on = 'RECEIVED_DATE', right_on = 'DECISION_DATE')

grp_month['backlog'] = (grp_month['n_x'] - grp_month['n_y']).cumsum()

fig, ax1 = plt.subplots(figsize=(10,5))

sns.scatterplot(grp_month[(grp_month['RECEIVED_DATE'].dt.year >= 2015)], x = 'RECEIVED_DATE', y = 'backlog', color = 'blue', ax = ax1)
plt.xticks(rotation = 45)
ax1.set_ylabel('Backlog', color = 'blue')
ax1.tick_params(axis = 'y',labelcolor='blue')

ax2 = ax1.twinx()
sns.scatterplot(grp_month[(grp_month['RECEIVED_DATE'].dt.year >= 2015)], x = 'RECEIVED_DATE', y = 'avg_processing_time_x', color = 'orange', ax = ax2)

sns.scatterplot(grp_month[(grp_month[grpvar].dt.year >= 2016)&(grp_month[grpvar].dt.year <= 2018)], x = grpvar, y = 'med_processing_time')
plt.xticks(rotation = 45)

sns.scatterplot(grp_month[(grp_month[grpvar].dt.year >= 2015)&(grp_month[grpvar].dt.year <= 2019)], x = grpvar, y = 'n')

sns.scatterplot(grp_month[(grp_month[grpvar].dt.year >= 2015)&(grp_month[grpvar].dt.year <= 2019)], x = grpvar, y = 'denial_rate')

# by week
grpvar = 'RECEIVED_DATE'
grp_week = perm.groupby([pd.Grouper(key = grpvar, freq = '1M'), 'fb']).agg(n = pd.NamedAgg('CASE_NUMBER', 'count'), avg_processing_time = pd.NamedAgg('processing_time', 'mean'), med_processing_time = pd.NamedAgg('processing_time', 'median'), denial_rate = pd.NamedAgg('denied', 'mean')).reset_index() 

samp = grp_week[grp_week[grpvar].between("2021-01-01",'2024-03-01')]
fig, ax1 = plt.subplots(figsize=(10,5))
sns.scatterplot(samp, x = grpvar, y = 'med_processing_time', ax = ax1, color = 'blue')
plt.xticks(rotation = 45)

ax2 = ax1.twinx()
sns.scatterplot(samp, x = grpvar, y = 'n', ax = ax2, color = 'orange')

# by month, color by year
sns.scatterplot(grp_month[(grp_month[grpvar].dt.year >= 2016)], x = 'rec_month')

# plotting by rec date
sns.histplot(perm, x = 'RECEIVED_DATE')


# by rec month
grp_rec_month = perm.groupby(['rec_month','rec_yr']).agg(n = pd.NamedAgg('CASE_NUMBER','count')).reset_index()

smf.ols("n ~ C(rec_month) + C(rec_yr)", grp_rec_month).fit().summary()

sns.histplot(perm[pd.isnull(perm['rec_month'])==0], x = 'rec_month')

## NEW: looking at PERM data 
perm = pd.read_excel(f"{root}/data/raw/perm/PERM_Disclosure_Data_FY2024_Q4.xlsx")
perm['processing_time'] = (perm['DECISION_DATE'] - perm['RECEIVED_DATE']).dt.days

# plotting distribution of case status by received and decision date
grp_rec = perm.groupby([pd.Grouper(key = 'RECEIVED_DATE', freq = '1M'),perm.CASE_STATUS]).agg(n = pd.NamedAgg('CASE_NUMBER','count')).reset_index()

grp_dec = perm.groupby([pd.Grouper(key = 'DECISION_DATE', freq = '1M'),perm.CASE_STATUS]).agg(n = pd.NamedAgg('CASE_NUMBER','count')).reset_index()

sns.scatterplot(grp_rec[grp_rec['RECEIVED_DATE'].dt.year > 2021], x = 'RECEIVED_DATE', y = 'n', hue = 'CASE_STATUS')
plt.xticks(rotation = 45)

# graph volume within one year (seasonal/sharp changes?)
sns.histplot(perm[perm['RECEIVED_DATE'].dt.year >= 2022], x = 'RECEIVED_DATE')

# within same 'submission batch' (same employer/received date) is there any variation in decision date?
perm['n_batch'] = perm.groupby(['RECEIVED_DATE','EMPLOYER_NAME'])['CASE_NUMBER'].transform('count')
perm['n_batch_dec'] = perm.groupby(['RECEIVED_DATE','EMPLOYER_NAME','DECISION_DATE'])['CASE_NUMBER'].transform('count')

## MISC QUESTIONS
perm_samp = perm[perm['RECEIVED_DATE']>='2016-01-01']
perm_samp['rec_date_num'] = perm_samp['RECEIVED_DATE'].map(pd.to_datetime).map(lambda d: d.toordinal())
perm_samp['dec_date_num'] = perm_samp['DECISION_DATE'].map(pd.to_datetime).map(lambda d: d.toordinal())

x = perm_samp[perm_samp['processing_time']>2].groupby(['RECEIVED_DATE','CASE_STATUS']).agg(decdate = pd.NamedAgg('DECISION_DATE','mean'), mindecdate = pd.NamedAgg('DECISION_DATE','min')).reset_index()

sns.scatterplot(x[(x['RECEIVED_DATE'].between('2017-01-01','2017-03-01'))&(x['decdate']<'2018-01-01')], x='RECEIVED_DATE',y = 'decdate', hue = 'CASE_STATUS')
lims = [
    min(plt.xlim()[0], plt.ylim()[0]),
    max(plt.xlim()[1], plt.ylim()[1])
]
plt.plot(lims, lims, 'r-', alpha=0.7)  # green line

# is processing sequential? (plot decision date against received date)
sns.regplot(perm_samp, x = 'rec_date_num', y = 'dec_date_num', x_bins = 100)
locs, labels = plt.xticks()
plt.xticks(locs, labels = [pd.to_datetime(int(t), origin="julian", unit="D").strftime("%Y-%m-%d") for t in locs], rotation=45)

sns.scatterplot(perm_samp, x = 'RECEIVED_DATE', y = 'DECISION_DATE')