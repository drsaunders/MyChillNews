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

#%%
def compute_sis_for_all(engine):
    sql_query = 'SELECT * FROM srcs;'
    srcs = pd.read_sql_query(sql_query,engine)
    src_lookup = {a.prefix:a.loc['index'] for i,a in srcs.iterrows()}
    
    headline_model = pickle.load( open( '../headline_model.pickle', "rb" ) )
    #headline_model = {'estimator':clf, 'vectorizer':headline_vectorizer, 'tfidf':headline_tfidf}
    
    sql_query = "SELECT * FROM frontpage" # WHERE fp_timestamp='%s' AND article_order <=10" % fp_timestamp
    frontpage_data =  pd.read_sql_query(sql_query,engine)
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
    
    # Should probably drop duplicates before doing z score or pct
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
