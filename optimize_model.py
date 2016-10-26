#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Workspace where I rapidly optimized the FB model and settled on the one used in
model_fb_reactions.

Created on Thu Sep 29 16:20:28 2016

@author: dsaunder
"""
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
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
import scipy.stats
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder    


def plot_fit_heatmap(real_y, estimate_y, vmax=100, cmap='Greens',reaction='angry',bins=[]):
    """Make a heat map showing the quality of the fit.
    
    Args:
        real_y: Array of the real values
        estimate_y: Estimated values to compare.
        vmax: Maximum value for the colour scale
        cmap: Which colour scheme to use (matplotlib colour maps)
        reaction: Which this is trying to model (sad or angry)
        bins: Bin edges for the 2D histogram
    """
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
    
def print_feature_importances(clf):
    """Output the importance of different words or bigrams, after random forest.
    Assumes revocab, the mapping from index to word, has been defined.
    
    Args:
        clf: A fitted random forest estimator.
    """
    ordering = np.argsort(clf.feature_importances_)
    ordering = ordering[::-1]
    words = [revocab[a] for a in ordering]
    for i in range(20):
        print "%.2f\t%s" % (clf.feature_importances_[ordering[i]], words[i])


def clean_fb_statuses(fb):
    """Apply a number of rules to remove facebook statuses and clean headlines.
    
    As we go, print out how many headlines are retained at each stage.
    
    Args:
        fb: Dataframe containing info about facebook statuses.
        
    Returns:
        The cleaned version of fb - fewerer statuses (usually about 45% less) and
        some strings removed from headlines.
    """
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


# Load the facebook status data
dbname = 'frontpage'
username = 'dsaunder'
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = 'SELECT * FROM fb_statuses;'
fb = pd.read_sql_query(sql_query,engine)
print len(fb)
fb = clean_fb_statuses(fb)

# Compute the proportion angry/sad/controversial
fb.loc[:,'prop_angry'] = fb.num_angries/fb.num_reactions
fb.loc[:,'prop_sad'] = fb.num_sads/fb.num_reactions
fb.loc[:,'prop_contro'] = fb.num_comments/fb.num_reactions

(fb, fb_test) = cross_validation.train_test_split(fb, test_size=0.2, random_state=0)
print len(fb)

#%% 
# Starting model. Baseline 
from sklearn.feature_extraction.text import CountVectorizer

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Vectorizer
headline_vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1,2), max_df=0.75)
bag = headline_vectorizer.fit_transform(fb.link_name)
revocab = {headline_vectorizer.vocabulary_[a]:a for a in headline_vectorizer.vocabulary_.keys()}

# TF-IDF transform
headline_tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = bag #headline_tfidf.fit_transform(bag)
y = np.log(fb.prop_angry+0.01)

# Start by just using the average of the proportion sad and angry reactions as
# the dependent.
y_sis = (fb.prop_angry + fb.prop_sad)/2

# Add src as a feature
sql_query = 'SELECT * FROM srcs;'
srcs = pd.read_sql_query(sql_query,engine)
src_lookup = {a.prefix:a.loc['index'] for i,a in srcs.iterrows()}
src_code = fb.src.map(src_lookup)
revocab[X.shape[1]]= 'SOURCE'
src_matrix = scipy.sparse.csr.csr_matrix(src_code.values.reshape(-1,1))
X_with_src = hstack((X, src_matrix))
src_encoder = OneHotEncoder()
src_hot = src_encoder.fit_transform(src_code.reshape(-1,1))
X_with_src2 = hstack((X, src_hot))
X_with_src2 = X_with_src2.tocsr()
      

##%%
## Fit random forest
# Commented out because it's slow
#clf = RandomForestRegressor(n_estimators=15, n_jobs=-1, verbose=1, oob_score=True)
#clf.fit(X_with_src,y)
#print "Oob score = %.3f" % clf.oob_score_

X[:,X.shape[1]-len(srcs):]

#%%
# Ridge regression exploration. Tried values of alpha by hand
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
clf_r = Ridge(alpha=5, normalize=True)

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2, y, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)

clf_r.fit(X_with_src2,y)
print [revocab[a] for a in np.argsort(clf_r.coef_)[-100:] if a < np.max(revocab.keys())]
#preds = cross_validation.cross_val_predict( clf_r, X_with_src2, y, cv=cv, n_jobs=-1, verbose=1)
#plt.figure()
#plt.plot(y,preds, '.')
#%%
#clf_rcv = RidgeCV(normalize=True, store_cv_values=True)
#clf_rcv.fit(X_with_src2,y)
#scores = cross_validation.cross_val_score( clf_rcv, X_with_src2, y, cv=cv, n_jobs=-1, scoring='r2', verbose=1)

#%%
# Lasso regression exploration. SLOW
from sklearn.linear_model import Lasso
clf_l = Lasso(alpha=0.0001, normalize=True)

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_l, X_with_src2, y, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)

clf_l.fit(X_with_src2,y)
print([revocab[a] for a in np.argsort(clf_l.coef_)[-100:]])

#%%
# ElasticNet exploration

from sklearn.linear_model import ElasticNet
clf_e = ElasticNet(alpha=0.0001, normalize=True)

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_e, X_with_src2, y, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)

clf_e.fit(X_with_src2,y)
[revocab[a] for a in np.argsort(clf_e.coef_)[-100:]]

 
#%%
# Nearest neighbour exploration

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=100)
scores = cross_validation.cross_val_score( neigh, X_with_src2, y, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)

#%%
# More in depth Ridge regression

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
clf_r = Ridge(alpha=10, normalize=True)

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2, y_sis, cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)
#%%
# Try log transform of the dependent 

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2, np.log1p(y_sis), cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)
plot_fit_heatmap(np.log1p(y_sis), preds, bins=np.arange(0,0.5,0.02),vmax=100)

#%%
# Try sqrt of dependent

clf_r = Ridge(alpha=2, normalize=True)
cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2, np.sqrt(y_sis), cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)

# This looks good, so scatter plot
preds = cross_validation.cross_val_predict( clf_r, X_with_src2, np.sqrt(y_sis), cv=cv, n_jobs=-1, verbose=1)
plt.figure()
#plt.plot(np.sqrt(y_sis),preds, '.')
sns.regplot(np.sqrt(y_sis),preds)
plot_fit_heatmap(np.sqrt(y_sis), preds, bins=np.arange(0,0.8,0.05),vmax=150)

#%%
# Try cube root
cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2, y_sis**(1./3), cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)
plot_fit_heatmap( y_sis**(1./3), preds, bins=np.arange(0,0.5,0.05),vmax=200)
#%%
clf_r = Ridge(alpha=5, normalize=True)
nz = np.nonzero(y_sis > 0)[0]
cv = cross_validation.KFold(len(nz), n_folds=5, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score( clf_r, X_with_src2[nz,:], y_sis.values[nz]**(1./3), cv=cv, n_jobs=-1, scoring='r2', verbose=1)
print np.mean(scores)
preds = cross_validation.cross_val_predict( clf_r,  X_with_src2[nz,:], y_sis.values[nz]**(1./3), cv=cv, n_jobs=-1, verbose=1)
plot_fit_heatmap( y_sis.values[nz]**(1./3), preds, bins=np.arange(0,0.5,0.05),vmax=200)



#%%%%%%% 
# LDA investigation

from gensim import matutils
from gensim.models.ldamodel import LdaModel

def print_topics(lda, vocab, n=10):
    """ Print the top words for each topic. """
    topics = lda.show_topics(num_topics=20, formatted=False)
    for ti, topic in enumerate(topics):
        print 'topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[0], t[1]) for t in topic[1]))

def fit_lda(X, vocab, num_topics=100, passes=20):
    """ Fit LDA from a scipy CSR matrix (X). """
    print 'fitting lda...'
    return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
                    passes=passes,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))
    print (time.time()-start)/60.


vocab = headline_vectorizer.get_feature_names()
start =time.time()
lda = fit_lda(X, vocab)
print (time.time()-start)/60.
print_topics(lda, vocab)


plt.plot(b[:,0],b[:,1],'.')



#%%
# Tentative attempt to use PCA
from sklearn.decomposition import PCA
start =time.time()
pca = PCA(n_components=1000)
pca.fit(X.toarray())
print (time.time()-start)/60.


#%%
# Grid search over several hyperparameters using ridge regression
# Actually did this a couple of times zooming in on the parameters by hand.
y_cube = y_sis.values**(1./3)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer

def SrcAdder(myX):
    print type(myX)
    return myX #hstack((myX, src_matrix)).to_csr()

pipeline = Pipeline([
    ('vect', CountVectorizer()),
#    ('add_srcs',FunctionTransformer(SrcAdder)),
    ('tfidf', TfidfTransformer()),
    ('clf', Ridge(alpha=10, normalize=True)),
])

parameters = {
    'vect__max_df': (0.62, 0.75, 0.87),
#    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.5,1,3),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(fb.link_name, y_cube)


