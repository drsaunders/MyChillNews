#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 22:29:14 2016

@author: dsaunder
"""
import pandas as pd 
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import cross_validation

le = preprocessing.LabelEncoder()

fb_tweet_comparison = pd.read_csv('fb_tweet_comparison.csv', encoding='utf-8') 


features = ['sis','src_x']
y = fb_tweet_comparison.prop_angry_fb
X = np.transpose(np.array([fb_tweet_comparison.sis, le.fit_transform(fb_tweet_comparison.src_x)]))
plt.plot(y,y_p,'.')

forest = RandomForestRegressor(n_estimators=150, n_jobs=-1)
forest.fit(X, y)
y_p = forest.predict(X)

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True)
scores = cross_validation.cross_val_score( forest, X, y, cv=cv, n_jobs=-1, scoring='r2')
y_cv = cross_validation.cross_val_predict( forest, X, y, cv=cv, n_jobs=-1)
#%%
plt.figure()
plt.scatter(y,y_cv)
#%%
# Using raw numbers of tweets and statuses to predict raw number of angry

features = ['sis','src_x']
y = fb_tweet_comparison.num_angry
X = np.transpose(np.array([fb_tweet_comparison.num_neg, fb_tweet_comparison.num_tweets, fb_tweet_comparison.num_reactions, le.fit_transform(fb_tweet_comparison.src_x)]))
forest = RandomForestRegressor(n_estimators=150, n_jobs=-1)
forest.fit(X, y)
y_p = forest.predict(X)
plt.figure()
plt.scatter(y,y_p)

cv = cross_validation.KFold(X.shape[0], n_folds=5, shuffle=True)
scores = cross_validation.cross_val_score( forest, X, y, cv=cv, n_jobs=-1, scoring='r2')
y_cv = cross_validation.cross_val_predict( forest, X, y, cv=cv, n_jobs=-1)
plt.figure()
plt.scatter(y,y_cv)


#
#
## Create linear regression object
#regr = linear_model.LinearRegression()
#
## Train the model using the training sets
#regr.fit(diabetes_X_train, diabetes_y_train)


