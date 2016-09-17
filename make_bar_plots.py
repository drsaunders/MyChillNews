#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:13:03 2016

@author: dsaunder
"""

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import codecs
from scipy.stats import beta
from scipy.special import gamma as gammaf
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#timestamp = '2016-09-13-0730'
timestamp = '2016-09-15-0722'
frontpagedir = '../frontpages/%s/' % timestamp

dbname = 'frontpage'
username = 'dsaunder'

sns.set(font_scale=1.4)
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM frontpage JOIN srcs ON frontpage.src=srcs.prefix WHERE fp_timestamp = '%s' AND article_order <=10;" % timestamp
frontpage_data = pd.read_sql_query(sql_query,engine)
#%%
for src in  np.unique(frontpage_data.src):
    
    plt.figure(figsize=(12,4))
    sns.barplot(x=frontpage_data.loc[frontpage_data.src == src,'sis'].values, y=range(10), orient='h', color='darkred')
    headlines = [re.sub('<[^>]*>','',a[:60]) for a in frontpage_data.loc[frontpage_data.src == src,'headline']]
    plt.yticks(range(10), headlines)
    plt.title('%s (Mean = %.2f)' % (frontpage_data.loc[frontpage_data.src == src,'name'].iloc[0], np.mean(frontpage_data.loc[frontpage_data.src == src,'sis'].values)))
    plt.xlim([0,0.35])
    plt.xlabel('Stress impact score')
    plt.tight_layout()
    plt.tight_layout()

    #%%
plt.figure()
sns.distplot(frontpage_data.sis,kde=False,bins=np.arange(0,0.4,0.05), color='indigo')    
plt.xlabel('Stress impact score')
plt.ylabel('Occurrences')

#%%
plt.figure()
sns.distplot(frontpage_data.num_tweets,kde=False,bins=np.arange(0,1400,50), color='blue')    
plt.xlabel('Number of tweets for article')
plt.ylabel('Occurrences')
#%%
plt.figure()
sns.distplot(frontpage_data.num_neg_tweets,kde=False, color='violet')    
plt.xlabel('Number of negative tweets for article')
plt.ylabel('Occurrences')


