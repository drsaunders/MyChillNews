#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 08:59:55 2016

@author: dsaunder
"""
from gensim.models.word2vec import Word2Vec
import pandas as pd
import re
import time
import numpy as np
import pickle
from sqlalchemy import create_engine
import sys
from sklearn.metrics import pairwise


tweet_corpus = pd.read_csv('../training.1600000.processed.noemoticon.utf-8.csv', names=['polarity','id','date','query','user','text'], encoding='utf-8')
y = tweet_corpus.polarity==4

tweets = tweet_corpus.text
tweet_corpus = None
#%%
def cleanTweets(tweets):
    tweets = tweets.apply(lambda x: x.lower())

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

    #Remove linebreaks
    tweets = tweets.apply(lambda x: re.sub('\n',u'',x))


    tweets = [z.split() for z in tweets]
    return tweets
    
tweets = cleanTweets(tweets)

filehandler = open('/Users/dsaunder/FrontPage - development/cleaned_training_tweets.pickle', 'wb') 
pickle.dump(tweets, filehandler) 
filehandler.close()


#%%

tweets = pickle.load( open( '/Users/dsaunder/FrontPage - development/cleaned_training_tweets.pickle', "r" ) )

#%%
n_dim = 300
#Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10)

start = time.time()
imdb_w2v.build_vocab(tweets)
print time.time() - start
#%%
#Train the model over train_reviews (this may take several minutes)
start = time.time()
imdb_w2v.train(tweets)
print time.time() - start
#%%
filehandler = open('/Users/dsaunder/FrontPage - development/trained_tweet_model.pickle', 'wb') 
pickle.dump(imdb_w2v, filehandler) 
filehandler.close()
#%%
imdb_w2v = pickle.load( open( '/Users/dsaunder/FrontPage - development/trained_tweet_model.pickle', "rb" ) )

#%%
#%%
#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


from sklearn.preprocessing import scale
start = time.time()
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in tweets])
train_vecs = scale(train_vecs)
print time.time() - start
    #%%
filehandler = open('/Users/dsaunder/FrontPage - development/training_wordvecs.pickle', 'wb') 
pickle.dump(train_vecs, filehandler) 
filehandler.close()

#%%
#imdb_w2v = pickle.load( open( '/Users/dsaunder/FrontPage - development/trained_tweet_model.pickle', "rb" ) )
#train_vecs =  pickle.load( open( '/Users/dsaunder/FrontPage - development/training_wordvecs.pickle', "rb" ) )

##%%
#
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#import seaborn as sns
#
#start = time.time()
#ts = TSNE(2)
#reduced_vecs = ts.fit_transform(train_vecs[::1000,:])
#print time.time() - start
##%%
#plt.figure()
#sns.set_palette('husl')
#plt.plot(reduced_vecs[np.nonzero(y[::1000])[0],0], reduced_vecs[np.nonzero(y[::100])[0],1],'.')
#plt.scatter(x=reduced_vecs[np.nonzero(np.invert(y[::1000]))[0],0], y=reduced_vecs[np.nonzero(np.invert(y[::100]))[0],1])

#%%
#Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
from sklearn.linear_model import SGDClassifier



lr = SGDClassifier(loss='log', penalty='l1')


lr.fit(train_vecs, y)

#%%
filehandler = open('/Users/dsaunder/FrontPage - development/tweet_classifier.pickle', 'wb') 
pickle.dump(lr, filehandler) 
filehandler.close()

##%%
#from sklearn import cross_validation
##
#cv = cross_validation.KFold(train_vecs.shape[0], n_folds=3, shuffle=True)
#scores = cross_validation.cross_val_score( lr, train_vecs, y, cv=cv, n_jobs=-1)
##
##%%
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10,n_jobs=-1, verbose=1)
#clf.fit(train_vecs,y)
#cv = cross_validation.KFold(train_vecs.shape[0], n_folds=2, shuffle=True)
#scores = cross_validation.cross_val_score( clf, train_vecs, y, cv=cv, n_jobs=-1)
##
#
#

#%%
tweets = None
train_vecs = None
y = None

dbname = 'frontpage'
username = 'dsaunder'
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

sql_query = "SELECT * FROM tweets WHERE fp_timestamp IS NOT NULL"
tweets_to_score =  pd.read_sql_query(sql_query,engine)
sql_query = "SELECT * FROM frontpage"
frontpage_data =  pd.read_sql_query(sql_query,engine)
tweets_to_score.loc[:,'article_id'] = [a.fp_timestamp+"-"+a.src+"-"+str(int(a.article)) for i,a in tweets_to_score.iterrows()]
frontpage_data.loc[:,'article_id'] = [a.fp_timestamp+"-"+a.src+"-"+str(int(a.article_order)) for i,a in frontpage_data.iterrows()]
tweets_to_score = tweets_to_score.merge(frontpage_data, on='article_id', how='left', suffixes=('','_r'))


def remove_headline_words(tweets, headlines):
    headlines = headlines.fillna('')
    headlines = cleanTweets(headlines)
    
    for i in range(len(tweets)):
        tweets[i] = [a for a in tweets[i] if a not in headlines[i]]

    return tweets
 
    
tweets = cleanTweets(tweets_to_score.text)
tweets = remove_headline_words(tweets, tweets_to_score.headline)

word_threshold = 4
#%%
real_tweet_vecs = np.concatenate([buildWordVector(z, n_dim) for z in tweets])
use_tweet  = [len(z) >= 4 for z in tweets]
#%%
negativity = lr.predict_proba(real_tweet_vecs)[:,0]


#%%

# ALTERNATE negativity computation
#negativity = pairwise.cosine_similarity(imdb_w2v['angry'].reshape(1,-1), real_tweet_vecs)
    
#negativity = (negativity.squeeze() > 0.65).astype(float)
tweet_negativity = pd.DataFrame({'tweet_id:':tweets_to_score.id,
                            'article_id':tweets_to_score.article_id,
                           'src':tweets_to_score.src,
                           'url':tweets_to_score.url,
                            'negative':negativity.squeeze()})

tweet_negativity = tweet_negativity.loc[use_tweet,:]
#%%
tweet_negativity.to_sql('tweet_negativity_wordvec', engine, if_exists='replace')

import scipy.stats

tweets_to_score.loc[:,'negative'] = negativity
summarized_negativity = pd.DataFrame(tweets_to_score.groupby('article_id').mean()['negative'])
summarized_negativity = summarized_negativity.merge(frontpage_data,left_index=True,right_on='article_id')
zsis = scipy.stats.zscore(summarized_negativity.negative)

article_summaries = pd.DataFrame({'article_id':summarized_negativity.article_id, 
'url':summarized_negativity.url,
'sis':summarized_negativity.negative
})

article_summaries.to_sql('sis_for_articles_wordvec', engine, if_exists='replace')
