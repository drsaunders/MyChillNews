#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:42:49 2016

@author: dsaunder
"""
from scipy.stats import beta
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.special import gamma as gammaf

if __name__ == "__main__":

    dbname = 'frontpage'
    username = 'dsaunder'
    # prepare for database
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

    sql_query = """
    SELECT article_id, url,
        SUM(negative::int) AS num_neg, 
        AVG(negative::int) AS prop_neg, 
        COUNT(negative::int) AS num_tweets 
    FROM tweet_negativity 
    GROUP BY article_id, url;    
    """
    summarized_negativity =  pd.read_sql_query(sql_query,engine)


#    frontpage_data =  pd.read_sql_query(sql_query,engine)

    with_tweets = summarized_negativity.num_tweets > 0
#    a1, b1, loc1, scale1 = beta.fit(summarized_negativity.loc[with_tweets,'prop_neg'])  
    mean=np.mean(summarized_negativity.prop_neg)
    var=np.var(summarized_negativity.prop_neg,ddof=1)
    a1=mean**2*(1-mean)/var-mean
    b1=alpha1*(1-mean)/mean

    fitted=lambda x,a,b:gammaf(a+b)/gammaf(a)/gammaf(b)*x**(a-1)*(1-x)**(b-1) #pdf of beta
    xx=np.arange(0,0.5,0.001)
    plt.figure()
    plt.hist(summarized_negativity.prop_neg,bins=100,normed=True)
    plt.plot(xx,fitted(xx,a1,b1),'g')
    sis = (summarized_negativity.num_neg+a1)/(summarized_negativity.num_tweets+ a1 + b1)
    sis[summarized_negativity.num_tweets ==0] = 0
    notnan = np.array([((not np.isnan(a)) and (a != 0)) for a in sis])

    zsis = scipy.stats.zscore(sis[notnan])
    
    article_summaries = pd.DataFrame({'article_id':summarized_negativity.article_id, 
    'url':summarized_negativity.url,
    'sis':sis, 
    'zsis':zsis,
    'prop_neg':summarized_negativity.prop_neg, 
    'num_neg':summarized_negativity.num_neg, 
    'num_tweets':summarized_negativity.num_tweets, 
    })
    article_summaries.to_sql('sis_for_articles', engine, if_exists='replace')
    
    