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
timestamp = '2016-09-21-0842'
frontpagedir = '../frontpages/%s/' % timestamp

dbname = 'frontpage'
username = 'dsaunder'

sns.set(font_scale=1.4)
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM frontpage JOIN srcs ON frontpage.src=srcs.prefix JOIN sis_for_articles ON sis_for_articles.url = frontpage.url WHERE fp_timestamp = '%s' AND article_order <=10;" % timestamp
print sql_query
frontpage_data = pd.read_sql_query(sql_query,engine)
#%%
for src in  np.unique(frontpage_data.src):
    
    plt.figure(figsize=(12,4))
    plt.barh(width=frontpage_data.loc[frontpage_data.src == src,'sis'].values, bottom=range(len(frontpage_data.loc[frontpage_data.src == src,'sis'].values)), color='darkred')
    headlines = [re.sub('<[^>]*>','',a[:60]) for a in frontpage_data.loc[frontpage_data.src == src,'headline']]
    plt.yticks(np.arange(10)+0.5, headlines)
    plt.title('%s (Mean = %.2f)' % (frontpage_data.loc[frontpage_data.src == src,'name'].iloc[0], np.mean(frontpage_data.loc[frontpage_data.src == src,'sis'].values)))
    plt.xlim([0,0.16])
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


#%%
sql_query = "SELECT * FROM sis_for_articles"
sis_for_articles_wordvec = pd.read_sql_query(sql_query,engine)
plt.figure()
sns.distplot(sis_for_articles_wordvec.sis)
plt.xlabel('Stress Impact Score')