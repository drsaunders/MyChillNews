#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 18:35:28 2016

@author: dsaunder
"""
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import codecs

def load_negative_words():
    neg_wds = []
    with codecs.open('negative-words.txt','r', encoding='utf-8') as f:
        for line in f:
            if len(line) <= 1:
                continue
            if line[0] == ';':
                continue
            neg_wds.append(unicode(line.strip()))

    return neg_wds

def detect_negative_tweets(tweets, headline):
    # Make the headline the second to last row
    tweets = tweets.append(pd.Series(headline,index=[np.max(tweets.index.values)+1]))
    
    # Add the negative words as the last row
    all_neg_words = load_negative_words()
    all_neg_words = [a for a in all_neg_words if not '-' in a]
    
    neg_str = ' '.join(all_neg_words)
    tweets = tweets.append(pd.Series(neg_str,index=[np.max(tweets.index.values)+1]))
    
    # Remove the url
    tweets = tweets.apply(lambda x: re.sub('http.*',u'',x))
    
    #Remove twitter handles
    tweets = tweets.apply(lambda x: re.sub('@[^\s]+',u'',x))
    
    #Remove hashtags
    tweets = tweets.apply(lambda x: re.sub('#[^\s]+',u'',x))
    
    #Remove passages in quotes
    tweets = tweets.apply(lambda x: re.sub('".*"',u'',x))
    
    #Replace ampersands
    tweets = tweets.apply(lambda x: re.sub('&amp;',u'and',x))
    
    # Make lowercase
    tweets = tweets.apply(lambda x: x.lower())
    
    # Remove special characters
    tweets = tweets.apply(lambda x: re.sub(u'(\u2018|\u2019|‘|’|“|”)',u'',x))
    
    count = CountVectorizer(stop_words=None)
    bag = count.fit_transform(tweets)
    vocab = count.vocabulary_
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Get rid of the last two lines of the bag and tweets (which were used for 
    # utilities)
    wd_bag = bag[:-2,:]
    tweets = tweets.iloc[:-2]
    
    neg_cols = np.nonzero(bag[-1])[1]
    
    # Remove words that appeared in the headline
    headline_cols = np.nonzero(bag[-2])[1]
    wd_bag[:,headline_cols] = 0
    
    # Remove negative words that appeared more than 5 times
    num_occurrences = np.asarray(wd_bag[:, neg_cols].sum(0))[0]
    too_many_negs = neg_cols[num_occurrences > 5]
    wd_bag[:,too_many_negs] = 0
    
    
    
    # Find the columns corresponding to the negative words, and use those to 
    # identify negative tweets (having more than 0 negative words)
    neg_wordcount = wd_bag[:,neg_cols].sum(1)
    neg_wordcount = np.squeeze(np.asarray(neg_wordcount))
    tweet_negative = neg_wordcount > 0
    
    print "\n\n" + headline
    print "Tweets with negative content"
    neg_tweet_indices = np.nonzero([neg_wordcount > 0])[1]
    for row in neg_tweet_indices:
        neg_words = [inv_vocab[a] for a in np.intersect1d(neg_cols, np.nonzero(wd_bag[row,:])[1])]
        print "TWEET %d: %s NEGATIVES: %s" % (row, tweets.iloc[row], ' '.join(neg_words))

    return tweet_negative
#%%
timestamp = '2016-09-12-0717'
frontpagedir = '../frontpages/%s/' % timestamp
#frontpage_data = pd.read_csv(timestamp + '_frontpage_data.csv', encoding='utf-8')
#fp_tweets =  pd.read_csv(timestamp + '_fp_tweets.csv', encoding='utf-8')

dbname = 'frontpage'
username = 'dsaunder'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM frontpage WHERE fp_timestamp = '%s';" % timestamp
frontpage_data = pd.read_sql_query(sql_query,engine,index_col='index')

sql_query = "SELECT * FROM tweets WHERE fp_timestamp = '%s';" % timestamp
fp_tweets =  pd.read_sql_query(sql_query,engine,index_col='index')

for src in np.unique(fp_tweets.src):
    for article in np.unique(fp_tweets.loc[fp_tweets.src == src,'article']):
        article_indicator= (frontpage_data.article_order == article) & (frontpage_data.src ==src)
        headline = frontpage_data.loc[article_indicator, 'headline'].values[0]
        negative_tweet = detect_negative_tweets(fp_tweets.loc[(fp_tweets.article == article) & (fp_tweets.src ==src),'text'], headline)
        frontpage_data.loc[article_indicator, 'num_tweets'] =  len(negative_tweet)
        frontpage_data.loc[article_indicator, 'num_neg_tweets'] = np.sum(negative_tweet)
        frontpage_data.loc[article_indicator, 'scaled_neg_tweets'] = float(np.sum(negative_tweet)) / len(negative_tweet)
        
    print src
    articles_with_tweets = frontpage_data.loc[(frontpage_data.src == src ) & np.invert(np.isnan(frontpage_data.scaled_neg_tweets))]
    print articles_with_tweets.sort_values('scaled_neg_tweets', ascending=False).loc[:,['headline','scaled_neg_tweets','num_tweets','num_neg_tweets','article_order']]

# Overwrite the current frontpage table
frontpage_data.to_sql('frontpage', engine, if_exists='replace')
