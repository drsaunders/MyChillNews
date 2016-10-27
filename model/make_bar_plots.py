#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Makes some nice figures for presentations.

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

dbname = 'frontpage'
username = 'dsaunder'

sns.set(font_scale=1.4)
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM frontpage JOIN srcs ON frontpage.src=srcs.prefix JOIN sis_for_articles ON sis_for_articles.url = frontpage.url WHERE fp_timestamp = '%s' AND article_order <=10;" % timestamp
print sql_query
frontpage_data = pd.read_sql_query(sql_query,engine)

#%%
# Bar graph of relative stress impacts of different headlines for different news sources
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
# Distribution of SIS 
plt.figure()
sns.distplot(frontpage_data.sis,kde=False,bins=np.arange(0,0.4,0.05), color='indigo')    
plt.xlabel('Stress impact score')
plt.ylabel('Occurrences')


#%%
# Colourful SIS distribution 

sql_query = "SELECT * FROM sis_for_articles_model"
sisfor = pd.read_sql_query(sql_query,engine)
plt.figure()
sns.distplot(sisfor.sis)
counts, bin_lefts = np.histogram(sisfor.sis,100)
pcts = [np.mean(a > sisfor.sis.values) for a in bin_lefts[:-1]]
color_range = np.array(sns.color_palette("coolwarm",n_colors=100))
bar_colors = color_range[np.floor(np.array(pcts)*100).astype(int)]

plt.figure()
sns.set_palette(bar_colors)
sns.barplot(bin_lefts[:-1], counts,linewidth=0)
plt.axis('off')
plt.xlabel('SIS score')
plt.savefig('sisdist.png')


#%%
# Fun code to look at proportion of sad and angry reactions over time. Requires
# Facebook statuses to be loaded into the dataframe fb.
#
##%%
#dts = [dateutil.parser.parse(a) for a in fb.status_published]
#fb.loc[:,'dt'] = dts
#fb.loc[:,'date'] = [a.date() for a in fb.dt]
#fb.loc[:,'datestr'] = [datetime.strftime(a,'%Y-%m-%d') for a in fb.date]
#
##%%
#gb = fb.groupby(['src','date'])
#anger = gb.mean()['prop_angry']
#sadness = gb.mean()['prop_sad']
#
##%%
#plt.figure()
#for the_src in np.unique(fb.src):
#    plt.plot(anger[the_src])
#plt.legend(np.unique(fb.src))
#plt.title('Anger')
#
##%%
#plt.figure()
#for the_src in np.unique(fb.src):
#    plt.plot(sadness[the_src])
#plt.legend(np.unique(fb.src))
#plt.title('Sad')
##%%
#plt.figure()
#
#gb2 = fb.groupby(['date'])
#plt.plot(gb2.mean()['prop_angry'],'.-')
#plt.plot(gb2.mean()['prop_sad'],'.-')
#plt.legend(['angry','sad'])
#
#fb.loc[fb.datestr=='2016-09-25','link_name']
