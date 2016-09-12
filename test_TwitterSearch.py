#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:29:53 2016

@author: dsaunder
"""

from TwitterSearch import *

try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords(['http://www.nytimes.com/2016/09/08/us/politics/obama-climate-change.html']) # let's define all words we would like to have a look for
    tso.set_language('en') # we want to see German tweets only
    tso.set_include_entities(False) # and don't give us all those entity information

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key = 'QJDuXCCt8y9MySeTXohxYyWD5',
        consumer_secret = 'VQrPaqDmGG0XRYFDeJcuwxrYQxjC9S0X9ZUEFSek6vYGLIsGgj',
        access_token = '559118063-UfrnluIuO44A9WqJDuoUnpEDrV8L72EIsDfJMqGq',
        access_token_secret = 'AXiYunWJQymYpO0r1s7zolgVXQwh2N0z4wc7bkOlQrtHl'
     )

#      this is where the fun actually starts :)
#%%
#    i = 1
#    for tweet in ts.search_tweets_iterable(tso):
#        print( '%d %s @%s tweeted: %s' % (i, tweet['created_at'], tweet['user']['screen_name'], tweet['text'] ) )
#        i = i+1

    
        #%%
#    all_tweets = list(ts.search_tweets_iterable(tso))
    todo = True
    next_max_id = 0
    i = 1
    all_tweets = []
    while todo:    
        response = ts.search_tweets(tso)
#    for i,tweet in enumerate(all_tweets['content']['statuses']):
#        print( '%d %s @%s tweeted: %s' % (i, tweet['created_at'], tweet['user']['screen_name'], tweet['text'] ) )
#    print( "Current rate-limiting status: %s" % ts.get_metadata()['x-rate-limit-reset'])
# check all tweets according to their ID
        all_tweets.extend(response['content']['statuses'])
        if len(all_tweets) > 0:
            next_max_id = all_tweets[-1]['id']
            next_max_id -= 1 # decrement to avoid seeing this tweet again
            
            # set lowest ID as MaxID
            tso.set_max_id(next_max_id)
        todo = not len(response['content']['statuses']) == 0
    #%%
    
    for i,tweet in enumerate(all_tweets):
        print( '%d %s @%s tweeted: %s' % (i, tweet['created_at'], tweet['user']['screen_name'], tweet['text'] ) )
        #%%
except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)