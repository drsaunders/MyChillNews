#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:36:27 2016

@author: dsaunder
"""
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
from scipy.stats import beta
import matplotlib.pyplot as plt
from scipy.special import gamma as gammaf
import numpy as np
import scipy

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM frontpage WHERE fp_timestamp = '%s' AND article_order <= 10;" % timestamp
frontpage_data = pd.read_sql_query(sql_query,engine,index_col='index')

frontpage_data = frontpage_data.fillna(0)


a1, b1, loc1, scale1 = beta.fit(frontpage_data.scaled_neg_tweets)  
fitted=lambda x,a,b:gammaf(a+b)/gammaf(a)/gammaf(b)*x**(a-1)*(1-x)**(b-1) #pdf of beta
xx=np.arange(0,0.5,0.001)
plt.plot(xx,fitted(xx,a1,b1),'g')


sis = (frontpage_data.num_neg_tweets+a1)/(frontpage_data.num_tweets+ a1 + b1)
frontpage_data.loc[:,'sis'] = sis
frontpage_data.loc[:,'zsis'] = scipy.stats.zscore(frontpage_data.loc[:,'sis'])
#%%
for src in  np.unique(fp_tweets.src):
    print '\n\n' + src
    print frontpage_data.loc[frontpage_data.src == src].sort_values('scaled_neg_tweets', ascending=False).loc[:,['headline','scaled_neg_tweets','sis','num_tweets','num_neg_tweets','article_order']]


frontpage_data.loc[frontpage_data.src == src].sort_values('scaled_neg_tweets', ascending=False).loc[:,['headline','scaled_neg_tweets','sis','num_tweets','num_neg_tweets','article_order']]


by_src = frontpage_data.loc[frontpage_data.num_tweets > 5,:].groupby('src')

by_src.mean().sort_values('scaled_neg_tweets', ascending=False)
#%%
ordered_by_scaled = by_src.mean().sort_values('scaled_neg_tweets', ascending=False).index.values
for i,src in enumerate(ordered_by_scaled):
    print str(i) + "\t" + src
    print frontpage_data.loc[frontpage_data.src == src].sort_values('scaled_neg_tweets', ascending=False).loc[:,['headline','scaled_neg_tweets','num_tweets','num_neg_tweets','article_order']]

#%%
ordered_by_sis = by_src.mean().sort_values('zsis', ascending=False).index.values
for i,src in enumerate(ordered_by_sis):
    print str(i+1) + "\t" + src
    print frontpage_data.loc[frontpage_data.src == src].sort_values('zsis', ascending=False).loc[:,['headline','zsis','sis','num_tweets','num_neg_tweets','article_order']]

#%%
ordered_by_sis = by_src.max().sort_values('sis', ascending=False).index.values
for i,src in enumerate(ordered_by_sis):
    print str(i) + "\t" + src
    print frontpage_data.loc[frontpage_data.src == src].sort_values('sis', ascending=False).loc[:,['headline','sis','num_tweets','num_neg_tweets','article_order']]

