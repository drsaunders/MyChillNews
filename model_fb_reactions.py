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

#%%


dbname = 'frontpage'
username = 'dsaunder'
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

sql_query = 'SELECT * FROM fb_statuses;'
fb = pd.read_sql_query(sql_query,engine)
print len(fb)

# Filter only links (more likely to also appear on the website)
fb = fb.loc[fb.status_type == 'link',:]
print len(fb)


# Remove items that aren't really links
lens = np.array([len(a) for a in fb.link_name])
fb = fb.loc[lens > 12,:]

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

# Compute the proportion angry/sad/controversial
fb.loc[:,'prop_angry'] = fb.num_angries/fb.num_reactions
fb.loc[:,'prop_sad'] = fb.num_sads/fb.num_reactions
fb.loc[:,'prop_contro'] = fb.num_comments/fb.num_reactions

# Create test split
(fb, fb_test) = cross_validation.train_test_split(fb, test_size=0.2, random_state=0)

#blank_src = [a is None for a in fb.src]

#%%
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
#%%
# Create the vectors of worsds
count = CountVectorizer(stop_words=stop, ngram_range=(1,1))

bag = count.fit_transform(fb.link_name)
revocab = {count.vocabulary_[a]:a for a in count.vocabulary_.keys()}
##%%
#from nltk.stem.porter import PorterStemmer
#
#porter = PorterStemmer()
#
#def tokenizer(text):
#    return text.split()
#def tokenizer_porter(text):
#    return [porter.stem(word) for word in text.split()]
#            
#%%
def tokenizer(text):
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(bag)
#%%
#X = bag
y = np.log(fb.prop_angry+0.01)
from sklearn.linear_model import LinearRegression
#clf = LinearRegression()
clf = RandomForestRegressor(n_estimators=15, n_jobs=-1, verbose=1)

cv = cross_validation.KFold(X.shape[0], n_folds=3, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf, X, y, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)
y_cv = cross_validation.cross_val_predict( clf, X, y, cv=cv, n_jobs=-1, verbose=1)
plt.figure()
sns.regplot(y,y_cv)
#%%
h = np.histogram2d(x=y,y=y_cv,bins=np.arange(-5,-0.5,0.5))
plt.figure()
sns.heatmap(h[0], annot=True,vmin=0, vmax=100, fmt='.0f')
plt.yticks(xt,h[2])
plt.gca().invert_yaxis()
xt = plt.xticks()[0]
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.xticks(xt,h[1])
#%%
clf.fit(X,y)
def print_feature_importances(clf):
    ordering = np.argsort(clf.feature_importances_)
    ordering = ordering[::-1]
    words = [revocab[a] for a in ordering]
    for i in range(20):
        print "%d\t%.2f\t%s" % (i+1, clf.feature_importances_[ordering[i]], words[i])
print_feature_importances(clf)
        #%%
#%%

y_binary = fb.prop_angry > 0
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15, n_jobs=-1, verbose=1,class_weight='balanced')
cv = cross_validation.KFold(X.shape[0], n_folds=3, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf, X, y_binary, cv=cv, n_jobs=-1)
print np.mean(scores)

clf.fit(X,y_binary)

print_feature_importances(clf)


#%%
g = sns.FacetGrid(fb.loc[fb.src.isin(['fox','cnn','nbc']),:], row="src")
g.map(sns.distplot,'prop_angry')

#%%
# Attempt at naive bayesian on classification data 

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X.toarray(),y_binary)

scores = cross_validation.cross_val_score( gnb, X.toarray(), y_binary, cv=5, n_jobs=-1)

#print gnb.score(X.toarray(),y_binary)s

#%% 
# 
plt.figure()
sns.distplot(fb.prop_angry,kde=False)
#%%
#plt.figure()
#sns.distplot(fb.prop_angry**(1/4.),kde=False)

y_root = fb.prop_angry# ** (1/4.)
y_root = y_root[y_root > 0]
y_root = np.log(y_root)
X_root = X[np.where(fb.prop_angry)[0],:]
from sklearn.linear_model import LinearRegression
#clf = LinearRegression()
clf = RandomForestRegressor(n_estimators=15, n_jobs=-1, verbose=1)

cv = cross_validation.KFold(X_root.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf, X_root, y_root, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)
y_cv = cross_validation.cross_val_predict( clf, X_root, y_root, cv=cv, n_jobs=-1, verbose=1)
plt.figure()
sns.regplot(y_root,y_cv)

#%%
h = np.histogram2d(x=y_root,y=y_cv)
plt.figure()
sns.heatmap(h[0], annot=True,vmin=0, vmax=100, fmt='.0f')
plt.gca().invert_yaxis()
xt = plt.xticks()[0]
plt.yticks(xt,h[2])
plt.xlabel('Actual value')
plt.ylabel('Predicted value')
plt.xticks(xt,h[1])


