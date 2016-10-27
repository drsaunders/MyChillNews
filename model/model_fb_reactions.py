#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains the model on the stored Facebook reaction data and headlines.

Uses a ridge regression model to map headlines (as 1-gram and 2-gram bags of
words) to SISs - stress impact scores, the square root of the mean of number
of sad and angry reactions. The effect of running this code is to train a brand
new model, save it as a pickle, and output some performance statistics.

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
from sklearn import cross_validation
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder    

import scipy.stats

#%%

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
    
def clean_fb_statuses(fb):
    """Apply a number of rules to remove facebook statuses and clean headlines.
    
    As we go, print out how many headlines are retained at each stage.
    
    Args:
        fb: Dataframe containing info about facebook statuses.
        
    Returns:
        The cleaned version of fb - fewerer statuses (usually about 45% less) and
        some strings removed from headlines.
    """
    # Filter to only links (more likely to also appear on the website)
    fb = fb.loc[fb.status_type == 'link',:]
    print len(fb)
    
    # Remove items that aren't really links
    lens = np.array([len(a) for a in fb.link_name])
    fb = fb.loc[lens > 12,:]
    
    #Remove items with no src
    src_is_none = [(a is None) or (a == np.nan) for a in fb.src]
    fb = fb.loc[np.invert(src_is_none),:]
    fb.loc[fb.src=='nbd','src'] = 'nbc'
    
    #Remove nuisance items that aren't really news stories, using regular expressions 
    nuisance_regexes = ['Take the quiz','Instagram photo by New York Times Archives','Your .* Briefing','Yahoo Movies','Yahoo Sports','Yahoo Movies UK','Yahoo UK & Ireland','Yahoo Finance','Yahoo Canada','Yahoo Music','Yahoo Celebrity','Yahoo Style + Beauty','The 10-Point.','Daily Mail Australia','USA TODAY Money and Tech']
    found= np.zeros(len(fb))
    for regex in nuisance_regexes:
        found = found + np.array([not re.search(regex, a) is None for a in fb.link_name])
    fb = fb.loc[np.invert(found.astype(bool)),:]
    
    print len(fb)

    # Remove items with no reactions at all (to prevent divide-by-zeros)
    fb = fb.loc[fb.num_reactions > 0,:]
    print len(fb)
    
    # Strip out extraneous phrases from headlines
    headline_regexes = [' - The Boston Globe',' \|.*$']
    for regex in headline_regexes:
        fb.loc[:,'link_name'] = [re.sub(regex, '',a) for  a in fb.loc[:,'link_name']]

    return fb

def upload_new_model():    
    """Upload the updated model to my web server, replacing the previous model.
    At the moment this is only run manually
    """
    execstr = 'scp -i ../../insight2016.pem ../../headline_model.pickle ubuntu@52.43.167.177:/home/ubuntu/' 
    print os.system(execstr)

    #%%
if __name__ == '__main__':

    # Read in collected FB statuses    
    dbname = 'frontpage'
    username = 'dsaunder'
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))    
    sql_query = 'SELECT * FROM fb_statuses;'
    fb = pd.read_sql_query(sql_query,engine)
    print len(fb)
    
    # Filter the statuses and clean the headlines
    fb = clean_fb_statuses(fb)
    
    # Compute the proportion angry/sad/controversial
    fb.loc[:,'prop_angry'] = fb.num_angries/fb.num_reactions
    fb.loc[:,'prop_sad'] = fb.num_sads/fb.num_reactions
    fb.loc[:,'prop_contro'] = fb.num_comments/fb.num_reactions
    
    # Create test split
    (fb, fb_test) = cross_validation.train_test_split(fb, test_size=0.2, random_state=0)
    
    #%%
    # Load stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    
    #%%
    # Tokenize the headlines and make into a bag of words

    headline_vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1,2), max_df=0.75)
    bag = headline_vectorizer.fit_transform(fb.link_name)
    X = bag 

    # Make a map back from word indexes to the english word
    revocab = {headline_vectorizer.vocabulary_[a]:a for a in headline_vectorizer.vocabulary_.keys()}

    sql_query = 'SELECT * FROM srcs;'
    srcs = pd.read_sql_query(sql_query,engine)
    src_lookup = {a.prefix:a.loc['index'] for i,a in srcs.iterrows()}
    src_code = fb.src.map(src_lookup)
    src_encoder = OneHotEncoder()
    src_hot = src_encoder.fit_transform(src_code.reshape(-1,1))
    X_with_src2 = hstack((X, src_hot))
    X_with_src2 = X_with_src2.tocsr()
      
    #%%
    # Create the dependent measure, my Stress Impact Score
    y = np.sqrt((fb.prop_angry + fb.prop_sad)/2)
    
    #%%
    # Cross validation check of the model
    from sklearn.linear_model import Ridge
    
    clf_r = Ridge(alpha=10, normalize=True)
    
    cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True, random_state=0)
    scores = cross_validation.cross_val_score( clf_r, X_with_src2, y, cv=cv, n_jobs=-1, verbose=1)
    print np.mean(scores)
    
    #%%
    # Test set performance
    
    clf_r.fit(X_with_src2, y)
    
    test_bag = headline_vectorizer.transform(fb_test.link_name)
    test_X = test_bag
    test_src = fb_test.src.map(src_lookup)
    test_src_hot = src_encoder.transform(test_src.reshape(-1,1))
    test_X= hstack((test_X, test_src_hot))
    test_y = np.sqrt((fb_test.prop_angry+fb_test.prop_sad)/2)
    
    print clf_r.score(test_X, test_y)
    
    # For debugging, a list of the words with the most positive coefficients
    [revocab[a] for a in np.argsort(clf_r.coef_)[-100:] if a < np.max(revocab.keys())]
    
    #%%
    # Save the regression model as a pickle, to be used to score the daily headlines.
    # At the moment I manually choose to upload it using upload_new_model().
    # Also the model is only trained on the training set - we could probably
    # do a bit better adding back the 20% of the test set.
    
    import pickle
    headline_model = {'estimator':clf_r, 'vectorizer':headline_vectorizer, 'src_encoder':src_encoder, 'fb_sis':y}
    filehandler = open('../../headline_model.pickle', 'wb')
    pickle.dump(headline_model, filehandler)
    filehandler.close()
    
    #%%
    # Check performance by looking at "front pages": when I had 10 news stories
    # on the same day from the same source in the test set, how did the model
    # do at predicting the stories overall, and then the average of the front page?

    import dateutil
    from datetime import datetime
    
    
    dts = [dateutil.parser.parse(a) for a in fb_test.status_published]
    fb_test.loc[:,'dt'] = dts
    fb_test.loc[:,'date'] = [a.date() for a in fb_test.dt]
    
    fb_test.loc[:,'sis'] = test_y
    fb_test.loc[:,'pred_sis'] = clf_r.predict(test_X)
    
    # For each date and front page, see if we have at least 10 facebook articles.
    # If so then store it in fpsamples.
    fpsamples = []
    fb_test.loc[:,'datestr'] = [datetime.strftime(a,'%Y-%m-%d') for a in fb_test.date]
    for d in np.unique(fb_test.datestr):
        for src in np.unique(fb_test.src):
            record = fb_test.loc[(fb_test.datestr==d) & (fb_test.src ==src),:].iloc[:10]

            if len(record) >= 10:
                fpsamples.append({'datestr':d, 'src':src, 'sis':np.mean(record.sis), 'pred_sis':np.mean(record.pred_sis)})
                       
    fpsamples = pd.DataFrame(fpsamples)
    #%%
    # Print R^2 for predictions of front pages as a whole.
    
    print scipy.stats.pearsonr(fpsamples.sis, fpsamples.pred_sis)[0]**2
    
    #%%
    # Make two scatter plots comparing real to predictions, either for individual
    # articles or the front page as a whole.
    pred_test_y  = clf_r.predict(test_X)
    plt.figure()
    sns.regplot(test_y, pred_test_y, line_kws={'color':sns.xkcd_palette(['dark grey'])[0]}, scatter_kws={"s": 20},color=sns.xkcd_palette(['orange'])[0])
    plt.xlabel('Actual article SIS')
    plt.ylabel('Predicted article SIS')
    plt.axis('square')
    plt.axis([0,0.7,0,0.7])
    plt.savefig('articlesis.png')
    
    plt.figure()
    sns.regplot(fpsamples.sis, fpsamples.pred_sis, line_kws={'color':sns.xkcd_palette(['dark grey'])[0]}, scatter_kws={"s": 60},color=sns.xkcd_palette(['purple'])[0])
    plt.xlabel('Actual mean SIS')
    plt.ylabel('Predicted mean SIS')
    plt.axis('square')
    plt.axis([0,0.5,0,0.5])
    plt.savefig('frontpagesis.png')


