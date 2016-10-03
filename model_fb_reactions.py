#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 07:58:38 2016

@author: dsaunder
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import scipy
import time
import matplotlib.pyplot as plt
import re
import os
from sklearn import preprocessing
from sklearn import cross_validation
from gensim.models.word2vec import Word2Vec
import tqdm
from scipy.sparse import hstack

import scipy.stats

#%%

def plot_feature_importance(features, fitted_forest):
    plt.figure()
    vals = fitted_forest.feature_importances_
    sortorder = np.flipud(np.argsort(vals))
    features = np.array(features)
    with sns.axes_style("whitegrid"):
        sns.stripplot(y=features[sortorder], x=vals[sortorder], orient="h", color='red', size=10)
    xl = plt.xlim()
    plt.xlim(0,xl[1])
    plt.grid(axis='y',linestyle=':')
    plt.xlabel('Feature importance score')

def print_feature_importances(clf):
    ordering = np.argsort(clf.feature_importances_)
    ordering = ordering[::-1]
    words = [revocab[a] for a in ordering]
    for i in range(20):
        print "%.2f\t%s" % (clf.feature_importances_[ordering[i]], words[i])

def plot_fit_heatmap(real_y, estimate_y, vmax=100, cmap='Greens',reaction='angry',bins=[]):
    if len(bins)==0:
        bins=np.arange(-5,0,0.5)
        
    h = np.histogram2d(x=real_y,y=estimate_y, bins=bins)
    plt.figure()

    sns.set(font_scale=1.5)
    sns.set_style()
    sns.heatmap(h[0], annot=False,vmin=0, vmax=vmax, fmt='.0f'
                ,cmap=cmap, cbar=False, square=True)
    plt.gca().invert_yaxis()
    plt.gca().set_frame_on(True)
    xt = plt.xticks()[0]
    plt.yticks(range(len(bins)),h[2][::-1])
    plt.xticks(range(len(bins)),h[2])
    plt.xlabel('Actual log proportion %s' % reaction)
    plt.ylabel('Predicted log proportion %s' % reaction)
    plt.tight_layout()
    
def clean_fb_statuses(fb):
    # Filter only links (more likely to also appear on the website)
    fb = fb.loc[fb.status_type == 'link',:]
    print len(fb)
    
    
    # Remove items that aren't really links
    lens = np.array([len(a) for a in fb.link_name])
    fb = fb.loc[lens > 12,:]
    
    #Remove items with no src
    src_is_none = [(a is None) or (a == np.nan) for a in fb.src]
    fb = fb.loc[np.invert(src_is_none),:]
    fb.loc[fb.src=='nbd','src'] = 'nbc'
    
    #Remove nuisance items that aren't really news stories
    nuisance_regexes = ['Take the quiz','Instagram photo by New York Times Archives','Your .* Briefing','Yahoo Movies','Yahoo Sports','Yahoo Movies UK','Yahoo UK & Ireland','Yahoo Finance','Yahoo Canada','Yahoo Music','Yahoo Celebrity','Yahoo Style + Beauty','The 10-Point.','Daily Mail Australia','USA TODAY Money and Tech']
    found= np.zeros(len(fb))
    for regex in nuisance_regexes:
        found = found + np.array([not re.search(regex, a) is None for a in fb.link_name])
    
    fb = fb.loc[np.invert(found.astype(bool)),:]
    
    print len(fb)
    
    
    # Remove items with no reactions at all (to prevent divide-by-zeros)
    
    fb = fb.loc[fb.num_reactions > 0,:]
    print len(fb)
    
    # Strip out extraneous phrases
    
    headline_regexes = [' - The Boston Globe',' \|.*$']
    for regex in headline_regexes:
        fb.loc[:,'link_name'] = [re.sub(regex, '',a) for  a in fb.loc[:,'link_name']]

    return fb
    #%%
#if __name__ == '__main__':

dbname = 'frontpage'
username = 'dsaunder'
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

sql_query = 'SELECT * FROM fb_statuses;'
fb = pd.read_sql_query(sql_query,engine)
print len(fb)

fb = clean_fb_statuses(fb)

# Compute the proportion angry/sad/controversial
fb.loc[:,'prop_angry'] = fb.num_angries/fb.num_reactions
fb.loc[:,'prop_sad'] = fb.num_sads/fb.num_reactions
fb.loc[:,'prop_contro'] = fb.num_comments/fb.num_reactions


# Create test split
(fb, fb_test) = cross_validation.train_test_split(fb, test_size=0.2, random_state=0)

#%%
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

#%%
# Create the vectors of worsds
from sklearn.preprocessing import OneHotEncoder

headline_vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1,2), max_df=0.75)
bag = headline_vectorizer.fit_transform(fb.link_name)
revocab = {headline_vectorizer.vocabulary_[a]:a for a in headline_vectorizer.vocabulary_.keys()}
X = bag 
sql_query = 'SELECT * FROM srcs;'
srcs = pd.read_sql_query(sql_query,engine)
src_lookup = {a.prefix:a.loc['index'] for i,a in srcs.iterrows()}
src_code = fb.src.map(src_lookup)
src_encoder = OneHotEncoder()
src_hot = src_encoder.fit_transform(src_code.reshape(-1,1))
X_with_src2 = hstack((X, src_hot))
X_with_src2 = X_with_src2.tocsr()



#%%
# Create the dependent measure
y = np.sqrt((fb.prop_angry + fb.prop_sad)/2)

#%%
# Cross validation check of our model
from sklearn.linear_model import Ridge

clf_r = Ridge(alpha=10, normalize=True)

start = time.time()
cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2, y, cv=cv, n_jobs=-1, verbose=1)
print np.mean(scores)
y_cv = cross_validation.cross_val_predict( clf_r, X_with_src2, y, cv=cv, n_jobs=-1, verbose=1)
print (time.time()-start)/60.


#%%
# Test set

clf_r.fit(X_with_src2, y)

test_bag = headline_vectorizer.transform(fb_test.link_name)
#test_X = headline_tfidf.transform(test_bag)
test_X = test_bag
test_src = fb_test.src.map(src_lookup)
test_src_hot = src_encoder.transform(test_src.reshape(-1,1))
test_X= hstack((test_X, test_src_hot))
test_y = np.sqrt((fb_test.prop_angry+fb_test.prop_sad)/2)

print clf_r.score(test_X, test_y)
#[revocab[a] for a in np.argsort(clf_r.coef_)[-100:] if a < np.max(revocab.keys())]

  #%%

#%%
import pickle
headline_model = {'estimator':clf_r, 'vectorizer':headline_vectorizer, 'src_encoder':src_encoder, 'fb_sis':y}
filehandler = open('../headline_model.pickle', 'wb')
pickle.dump(headline_model, filehandler)
filehandler.close()

#%%
# Front page analysis
import dateutil
from datetime import datetime


dts = [dateutil.parser.parse(a) for a in fb_test.status_published]
fb_test.loc[:,'dt'] = dts
fb_test.loc[:,'date'] = [a.date() for a in fb_test.dt]


fb_test.loc[:,'sis'] = test_y
fb_test.loc[:,'pred_sis'] = clf_r.predict(test_X)

fpsamples = []
fb_test.loc[:,'datestr'] = [datetime.strftime(a,'%Y-%m-%d') for a in fb_test.date]
for d in np.unique(fb_test.datestr):
    for src in np.unique(fb_test.src):
        record = fb_test.loc[(fb_test.datestr==d) & (fb_test.src ==src),:].iloc[:10]
#        print len(record)
        if len(record) >= 10:
            fpsamples.append({'datestr':d, 'src':src, 'sis':np.mean(record.sis), 'pred_sis':np.mean(record.pred_sis)})
                   #%%
fpsamples = pd.DataFrame(fpsamples)
#%%


print scipy.stats.pearsonr(fpsamples.sis, fpsamples.pred_sis)[0]**2

#%%
pred_test_y  = clf_r.predict(test_X)
plt.figure()
sns.regplot(test_y, pred_test_y, line_kws={'color':sns.xkcd_palette(['dark grey'])[0]}, scatter_kws={"s": 20},color=sns.xkcd_palette(['orange'])[0])
plt.xlabel('Actual article SIS')
plt.ylabel('Predicted article SIS')
plt.axis('square')
plt.axis([0,0.7,0,0.7])

pred_test_y  = clf_r.predict(test_X)
plt.figure()
sns.regplot(fpsamples.sis, fpsamples.pred_sis, line_kws={'color':sns.xkcd_palette(['dark grey'])[0]}, scatter_kws={"s": 60},color=sns.xkcd_palette(['purple'])[0])
plt.xlabel('Actual front page SIS')
plt.ylabel('Predicted front page SIS')
plt.axis('square')
plt.axis([0,0.5,0,0.5])

#%

