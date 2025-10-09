import wrds
import pandas as pd
import numpy as np
import re
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

db = wrds.Connection(username = 'amykimecon')

jobs_raw = db.raw_sql("SELECT a.job_id, company, country, post_date, remove_date, rcid, remote_type, title_raw, description FROM (SELECT * FROM revelio.postings_cosmos WHERE country = 'United States' AND role_k1500 = 'software engineering' AND source_indeed = True) AS a LEFT JOIN revelio.postings_cosmos_raw AS b ON a.job_id = b.job_id")

# write jobs_raw to parquet
jobs_raw.to_parquet(f'{root}/data/int/software_eng_postings_raw_17sep2025.parquet', index = False)

jobs_raw.to_csv('/Users/amykim/Downloads/jobs_raw.csv', index = False)

jobs = jobs_raw.copy().sample(1000)

# defining regex strings for sponsorship and eoe and security clearance
sponsor_str = 'sponsor|visa|citizen|immigration|work permit|h\\-?\\s?1b|h\\-?\\s?2b|\\sopt\\s|\\scpt\\s|green card|permanent resident|work eligibility|authorization'

eoe_str = 'equal\\-?opportunity|eoe|e\\.o\\.e\\.|affirmative action|aa/?\\s?eeo|protected|sex'

clearance_str = 'clearance|ts/?\\s?sci'

#TODO: filter out everify?

# cleaning description
jobs['description_clean'] = jobs['description'].str.replace('U\\.?\\s?S\\.?', 'US', regex = True).str.replace('\\.\\s?', '.\n', regex = True).str.replace('\\s(?=[^\\s]+\\:)', '\n', regex = True).str.replace('</?(div|ul|li)>',' ', regex = True)

# extracting text around sponsorship mention (up to 1 sentence before and after)
jobs['sponsor_text'] = jobs['description'].str.replace('U\\.?\\s?S\\.?', 'US', regex = True).str.findall(f'(?:\\.|\\\n)([^\\.]*(?:{sponsor_str})[^\\.]*(?:\\.|\\\n))', flags = re.IGNORECASE)

# indicator for whether job description mentions sponsorship at all
jobs['any_sponsorship'] = jobs['sponsor_text'].apply(lambda x: 1 if x else 0)

# indicator for whether sponsor text also mentions eoe
jobs['eoe'] = jobs.apply(lambda row: [1 if re.search(f'{eoe_str}', x, flags = re.IGNORECASE) is not None else 0 for x in row['sponsor_text']] if row['any_sponsorship'] == 1 else None, axis = 1)

# indicator for whether sponsor text also mentions security clearance
jobs['clearance'] = jobs.apply(lambda row: [1 if re.search(f'{clearance_str}', x, flags = re.IGNORECASE) is not None else 0 for x in row['sponsor_text']] if row['any_sponsorship'] == 1 else None, axis = 1)

# indicator for whether job description mentions sponsorship without eoe or clearance
jobs['clean_sponsorship'] = jobs.apply(lambda row: ((1-np.array(row['eoe'])) * (1-np.array(row['clearance']))).max() if row['any_sponsorship'] == 1 else 0, axis = 1)

# number of sponsorship text extracts
jobs['sponsor_n'] = jobs.apply(lambda row: len(row['sponsor_text']), axis = 1)

for jid in jobs[jobs['any_sponsorship']==1].sample(10)['job_id']:
    print(f'\njob id: {jid}')
    print(f'post date: {jobs[jobs["job_id"] == jid]["post_date"].values[0]}')
    print(f'title: {jobs[jobs["job_id"] == jid]["title_raw"].values[0]}')
    print(f'company: {jobs[jobs["job_id"] == jid]["company"].values[0]}')
    for i in range(jobs[jobs['job_id'] == jid]['sponsor_n'].values[0]):
        print(f"--Sponsor Text Extract #{i + 1}:--")
        print(f"----Clean indicator: {jobs[jobs["job_id"] == jid]["clean_sponsorship"].values[0]}")
        print(f"----EOE indicator: {jobs[jobs["job_id"] == jid]["eoe"].values[0][i]}")
        print(f"----Clearance indicator: {jobs[jobs["job_id"] == jid]["clearance"].values[0][i]}")
        print(jobs[jobs["job_id"] == jid]["sponsor_text"].values[0][i])
    # print(f'sponsor text: {jobs[jobs["job_id"] == jid]["sponsor_text"].values[0]}')
    # print(f'description: {jobs[jobs["job_id"] == jid]["description"].values[0]}')
    # print(f'description clean: {jobs[jobs["job_id"] == jid]["description_clean"].values[0]}')


for jid in jobs.sample(10)['job_id']:
    print(f'\njob id: {jid}')
    print(f'description: {jobs[jobs["job_id"] == jid]["description"].values[0]}')
    print(f'description clean: {jobs[jobs["job_id"] == jid]["description_clean"].values[0]}')

## TODO NEXT: plot number of mentions over time