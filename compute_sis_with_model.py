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
from sklearn.preprocessing import OneHotEncoder


def score_frontpage_frame(fp, src_lookup, headline_model):
    headlines = fp.headline
    bag = headline_model['vectorizer'].transform(headlines)
    X = bag #headline_model['tfidf'].transform(bag)
    src_code = fp.src.map(src_lookup)

    src_hot = headline_model['src_encoder'].transform(src_code.reshape(-1,1))
    X = hstack((X, src_hot))

    sis = headline_model['estimator'].predict(X)

    return sis

def score_phrase(text, src, headline_model):
    src_lookup = {u'abc': 10,
 u'bbc': 12,
 u'bos': 15,
 u'cnn': 7,
 u'dm': 9,
 u'fox': 3,
 u'goo': 5,
 u'gua': 14,
 u'huf': 6,
 u'lat': 2,
 u'nbc': 8,
 u'nyt': 0,
 u'usa': 13,
 u'wap': 4,
 u'wsj': 11,
 u'yahoo': 1}
    fp = pd.DataFrame({'headline':text, 'src':src},index=[0])
    phrase_sis = score_frontpage_frame(fp, src_lookup, headline_model)
    return phrase_sis[0]
#%%
def compute_sis_for_all(engine, suppress_db_write=False):
#%%
    sql_query = 'SELECT * FROM srcs;'
    srcs = pd.read_sql_query(sql_query,engine)
    src_lookup = {a.prefix:a.loc['index'] for i,a in srcs.iterrows()}

    headline_model = pickle.load( open( '../headline_model.pickle', "rb" ) )
    #headline_model = {'estimator':clf, 'vectorizer':headline_vectorizer, 'tfidf':headline_tfidf}

    print "Loading all articles..."
    sql_query = "SELECT * FROM frontpage" # WHERE fp_timestamp='%s' AND article_order <=10" % fp_timestamp
    frontpage_data =  pd.read_sql_query(sql_query,engine)
    print "Num headlines =  %d" % len(frontpage_data)
    print "Computing sis..."
    

    sis = score_frontpage_frame(frontpage_data, src_lookup, headline_model)

    # Should probably drop duplicates before doing z score or pct
    sis_pct = [np.mean(a > sis) for a in sis]

    zsis = scipy.stats.zscore(sis)
    article_summaries = pd.DataFrame({'article_id':frontpage_data.article_id
                                      ,'url':frontpage_data.url
                                      ,'sis':sis
                                      ,'zsis':zsis
                                      ,'sis_pct':sis_pct

                                     })

    if not suppress_db_write:
        print "Writing SIS to database..."
        article_summaries.to_sql('sis_for_articles_model', engine, if_exists='replace', chunksize=100)
    else:
        print "Suppressed writing SIS to database"


#%%
if __name__ == '__main__':
    dbname = 'frontpage'
    username = 'ubuntu'
    # prepare for database
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    compute_sis_for_all(engine)
