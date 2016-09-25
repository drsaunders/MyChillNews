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
from scipy.stats import beta
from scipy.special import gamma as gammaf
import scipy
from tqdm import *

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
    if len(tweets) == 0:
        return []
        
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
#    tweets = tweets.apply(lambda x: x.lower())
    
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
#    if len(neg_wordcount) == 1:
#        print neg_wordcount
#        return
#        tweet_negative = np.squeeze(neg_wordcount > 0)
#        print tweet_negative
#    else:
#        print neg_wordcount
    neg_wordcount = np.squeeze(np.asarray(neg_wordcount))
 
    tweet_negative = float(neg_wordcount > 0)
    if not hasattr(tweet_negative, '__iter__'):
        tweet_negative = np.array([tweet_negative])
        
    neg_words = []
    for row in range(len(tweets)):
        if tweet_negative[row]:
            neg_word_list = [inv_vocab[a] for a in np.intersect1d(neg_cols, np.nonzero(wd_bag[row,:])[1])]            
            neg_words.append(' '.join(neg_word_list))
        else:
            neg_words.append('')
                             #        print "%s NEGATIVES: %s" % (row, tweets.iloc[row] )#, ' '.join(neg_words))
#        print "TWEET %d:  %s" % (row, tweets.iloc[row] )#, ' '.join(neg_words))
        
    return tweet_negative, neg_words
#%%
if __name__ == "__main__":
    
    dbname = 'frontpage'
    username = 'dsaunder'
    # prepare for database
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

    sql_query = "SELECT * FROM tweets WHERE fp_timestamp IS NOT NULL;"
    tweets_to_score =  pd.read_sql_query(sql_query,engine)
    sql_query = "SELECT * FROM frontpage"
    frontpage_data =  pd.read_sql_query(sql_query,engine)
    tweets_to_score.loc[:,'article_id'] = [a.fp_timestamp+"-"+a.src+"-"+str(int(a.article)) for i,a in tweets_to_score.iterrows()]
    frontpage_data.loc[:,'article_id'] = [a.fp_timestamp+"-"+a.src+"-"+str(int(a.article_order)) for i,a in frontpage_data.iterrows()]
    #%%
#    article_id = '2016-09-15-0722-lat-1'
    tweet_negativity_list = []
    for article_id in tqdm(np.unique(tweets_to_score.article_id)):
        article_info = frontpage_data.loc[frontpage_data.article_id == article_id,:]
        if len(article_info) == 0:
            continue
        headline = article_info.headline.values[0]
        tweets_for_article = tweets_to_score.loc[tweets_to_score.article_id == article_id ,:]
        (negativity, neg_words) = detect_negative_tweets(tweets_for_article.text, headline)
        ntweets = len(tweets_for_article)
        new_scores = pd.DataFrame({'tweet_id:':tweets_for_article.id,
                                    'article_id':[article_id]*ntweets,
                                   'src':[article_info.src.values[0]]*ntweets,
                                   'url':[article_info.url.values[0]]*ntweets,
                                    'negative':negativity,
                                    'neg_words':neg_words})
        tweet_negativity_list.append(new_scores)

    tweet_negativity = pd.concat(tweet_negativity_list)
    
    tweet_negativity.to_sql('tweet_negativity', engine, if_exists='replace')
    #        tweet_negativity = pd.concat(tweet_negativity_list, ignore_index=True)
#    # Go through every news story and score it using the number of negative tweets
#    for fp
#    for src in np.unique(fp_tweets.src):
#        for article in np.unique(frontpage_data.loc[frontpage_data.src == src,'article_order']):
#            detect_negative_tweets(fp_tweets.loc[(fp_tweets.article == article) & (fp_tweets.src ==src),'text'], headline)
#            
#            article_indicator= (frontpage_data.article_order == article) & (frontpage_data.src ==src)
#            headline = frontpage_data.loc[article_indicator, 'headline'].values[0]
#            negative_tweet = detect_negative_tweets(fp_tweets.loc[(fp_tweets.article == article) & (fp_tweets.src ==src),'text'], headline)
#            frontpage_data.loc[article_indicator, 'num_tweets'] =  len(negative_tweet)
#            frontpage_data.loc[article_indicator, 'num_neg_tweets'] = np.sum(negative_tweet)
#            if len(negative_tweet) == 0:
#                frontpage_data.loc[article_indicator, 'scaled_neg_tweets'] = np.nan
#            else:
#                frontpage_data.loc[article_indicator, 'scaled_neg_tweets'] = float(np.sum(negative_tweet)) / len(negative_tweet)
#            
#        print src
#        articles_with_tweets = frontpage_data.loc[(frontpage_data.src == src ) & np.invert(np.isnan(frontpage_data.scaled_neg_tweets))]
#        print articles_with_tweets.sort_values('scaled_neg_tweets', ascending=False).loc[:,['headline','scaled_neg_tweets','num_tweets','num_neg_tweets','article_order']]
#    
#    
#    frontpage_data = add_sis_scores(frontpage_data)
#    
#    by_src = frontpage_data.loc[frontpage_data.num_tweets > 5,:].groupby('src')
#    #ordered_by_zsis = by_src.mean().sort_values('zsis', ascending=False)
#    
#    
#    # Overwrite the current frontpage table
#    frontpage_data.to_sql('frontpage', engine, if_exists='replace')



