#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:56:31 2016

@author: dsaunder
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import scipy
import time
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import hstack
import scipy.stats
import getpass

fp_timestamp= '2016-09-24-0948'

dbname = 'frontpage'
username = getpass.getuser()
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = 'SELECT * FROM srcs;'
srcs = pd.read_sql_query(sql_query,engine)
src_lookup = {a.prefix:a.loc['index'] for i,a in srcs.iterrows()}

headline_model = pickle.load( open( '../headline_model.pickle', "rb" ) )
#headline_model = {'estimator':clf, 'vectorizer':headline_vectorizer, 'tfidf':headline_tfidf}

sql_query = "SELECT * FROM frontpage" # WHERE fp_timestamp='%s' AND article_order <=10" % fp_timestamp
frontpage_data =  pd.read_sql_query(sql_query,engine)
frontpage_data.loc[:,'article_id'] = [a.fp_timestamp+"-"+a.src+"-"+str(int(a.article_order)) for i,a in frontpage_data.iterrows()]
headlines = frontpage_data.headline
bag = headline_model['vectorizer'].transform(headlines)
X = headline_model['tfidf'].transform(bag)
src = frontpage_data.src.map(src_lookup)
src_matrix = scipy.sparse.csr.csr_matrix(src.values.reshape(-1,1))
X= hstack((X, src_matrix))
#%%
angry_estimates = headline_model['angry_estimator'].predict(X)
sad_estimates = headline_model['sad_estimator'].predict(X)
sis = (angry_estimates + sad_estimates)/2

plt.figure()
sns.distplot(np.exp(sis)-0.01, kde=False)
print headlines.iloc[np.argsort(sis)]

sis_pct = [np.mean(a > sis) for a in sis]

article_summaries = pd.DataFrame({'article_id':frontpage_data.article_id
                                  ,'url':frontpage_data.url
                                  ,'sis':sis
                                  ,'zsis':scipy.stats.zscore(sis)
                                  ,'angry':angry_estimates
                                  ,'zangry':scipy.stats.zscore(angry_estimates)
                                  ,'sad':sad_estimates
                                  ,'zsad':scipy.stats.zscore(sad_estimates)
                                  ,'sis_pct':sis_pct

                                 })

article_summaries.to_sql('sis_for_articles_model', engine, if_exists='replace', chunksize=1000)


#%%
from wordcloud import WordCloud
def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    the_color = sns.color_palette("coolwarm",n_colors=2)[0]
    return (255,0,0)

def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    the_color = sns.color_palette("coolwarm",n_colors=2)[1]
    return (0,0,255)
width = 2000
height = 1600
x,y = np.meshgrid(range(width),range(height))
ellipse_mask = ((x-1000)/1000.)**2 + ((y-800)/800.)**2 > 1
ellipse_mask = ellipse_mask.astype(int)*255
    #%%

lower_text = ' '.join(headlines.iloc[np.argsort(sis)[:1000]])
wordcloud = WordCloud(mask=ellipse_mask, background_color='white', relative_scaling=1,width=2000,height=1600, min_font_size=30).generate(lower_text)
wordcloud.recolor(0,blue_color_func)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('lower.png')

#%%
upper_text = ' '.join(headlines.iloc[np.argsort(sis)[-1000:]])

wordcloud = WordCloud(mask=ellipse_mask, background_color='white', relative_scaling=1,width=2000,height=1600, min_font_size=30).generate(upper_text)
#wordcloud = WordCloud(background_color='white', relative_scaling=0.1, max_words=100).generate(upper_text)
wordcloud.recolor(0,red_color_func)
plt.figure(figsize=(10,10))
#wordcloud = WordCloud(background_color='white').generate(lower_text)
#plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('upper.png')
#%%
plt.figure()
plt.imshow(ellipse_mask)
