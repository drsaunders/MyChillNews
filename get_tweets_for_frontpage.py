#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:32:12 2016

@author: dsaunder
"""
import pandas as pd
from TwitterSearch import *
import numpy as np
import time
import dateutil.parser
from datetime import datetime, timedelta
from pytz import timezone
#%%
def get_reset_info(ts):
    try:
        print "Got here!"
        response = ts.search_tweets(tso)
    except TwitterSearchException as e:
        print e
        pass 
    
    reset_dt = datetime.utcfromtimestamp(int(response['meta']['x-rate-limit-reset'])) + timedelta(hours=-4)
    reset_time = reset_dt.time()
    print response['meta']['x-rate-limit-reset']
    print "Minutes till reset: %.2f" % ((int(response['meta']['x-rate-limit-reset']) - time.time())/60.)
    print "Time of reset: " + str(reset_time)
    print response['meta']['x-rate-limit-remaining']

#%%

def get_all_tweets_for_search(searchstr, ts):
    global total_num_searches
    all_tweets = []
    start_time = time.time()
    
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords([searchstr])
    tso.set_include_entities(False) # and don't give us all those entity information

    try:
        tso = TwitterSearchOrder() # create a TwitterSearchOrder object
        tso.set_keywords([searchstr])     
        tso.add_keyword("-filter:retweets") # This line eliminates all retweets
        tso.set_language('en') # we want to see English tweets only
        tso.set_include_entities(False) # and don't give us all those entity information
    
        print "\nSTARTING TWITTER SEARCH string: " + searchstr
        todo = True
        next_max_id = 0
        response_page = 1
        while todo:    
            response = ts.search_tweets(tso)
            total_num_searches = total_num_searches + 1
            all_tweets.extend(response['content']['statuses'])
            if len(response['content']['statuses']) > 0:
                print " Response page %d, %d tweets" % (response_page, len(response['content']['statuses'])) 

                next_max_id = all_tweets[-1]['id']
                next_max_id -= 1 # decrement to avoid seeing this tweet again
                
                # set lowest ID as MaxID
                tso.set_max_id(next_max_id)
                response_page = response_page + 1
            else:
                print " No more pages of response"
                todo = False
        if len(all_tweets) > 0:
            oldest_created_date = dateutil.parser.parse(all_tweets[-1]['created_at'])
            oldest_created_date = oldest_created_date.astimezone(timezone('US/Eastern'))
        
    except TwitterSearchException as e: # take care of all those ugly errors if there are some
        print "TWITTER SEARCH ERROR"
        print(e)
        total_num_searches = 0
        print "Current time: " + str(datetime.now().time())
        return all_tweets

    if len(all_tweets) > 0:
        oldest_created_date = dateutil.parser.parse(all_tweets[-1]['created_at'])
        oldest_created_date = oldest_created_date.astimezone(timezone('US/Eastern'))
        print "Total retrieved: %d" % len(all_tweets)
        print "Oldest tweet retrieved was created " + str(oldest_created_date)
      
    print "Time taken for twitter search: %.1f seconds" % ((time.time() - start_time)/1.)
    print "Number of searches left this time slice: " + response['meta']['x-rate-limit-remaining']
    return all_tweets
#%%
def score_negative_words(tweets):
    reactions = []
    for t in tweets:
        reactions.append(any(a in t for a in ['sad','awful','terrible','bad','sucks','unhappy','upset','angry','depressing']))

    return reactions
#%%
global total_num_searches 

try:
    ts = TwitterSearch(
        consumer_key = 'QJDuXCCt8y9MySeTXohxYyWD5',
        consumer_secret = 'VQrPaqDmGG0XRYFDeJcuwxrYQxjC9S0X9ZUEFSek6vYGLIsGgj',
        access_token = '559118063-UfrnluIuO44A9WqJDuoUnpEDrV8L72EIsDfJMqGq',
        access_token_secret = 'AXiYunWJQymYpO0r1s7zolgVXQwh2N0z4wc7bkOlQrtHl'
    )
except TwitterSearchException as e:
    print "TWITTER ERROR: Too many simultaneous connections"
    print e

#%%
# MAIN 
total_start_time = time.time()
total_num_searches = 0
timestamp = '2016-09-12-0717'
frontpage_data = pd.read_csv(timestamp + '_frontpage_data.csv')
fp_tweets = pd.DataFrame()
for src in ['lat']: #np.unique(frontpage_data.src):
    articles_for_paper = frontpage_data.loc[(frontpage_data.src ==src) & (frontpage_data.article_order <=10),: ]
    for i in range(0,len(articles_for_paper)):
        print 'Searches so far: %d. Article %d for source %s. ' % (total_num_searches, i+1, src)
        url = articles_for_paper.iloc[i,:].url
        if src == 'lat':
            url = re.sub('http://','http:/',url)
        tweets_for_article = get_all_tweets_for_search(url, ts)
#    for article in articles_for_paper.iterrows():
#        tweets_for_article = get_all_tweets_for_search(article.url)
        for j,tweet in enumerate(tweets_for_article):
            fp_tweets = fp_tweets.append({'src':src,                                         
            'fp_timestamp':timestamp,
            'article':i+1,
            'order':j, 
            'created':tweet['created_at'],
            'user':tweet['user']['screen_name'],
            'text':tweet['text'],
            'retweet_count':tweet['retweet_count']
                },
                ignore_index=True)
            
            #%%

fp_tweets.to_csv(timestamp + '_fp_tweets.csv',index=False, encoding='utf-8')
print "Total time elapsed: %.1f minutes." % ((time.time() - total_start_time ) / 60.)

#fp_twt = pd.read_csv('2016-09-08-1518_fp_tweets2.csv')
##%%
#fp_twt.loc[:,'negative'] = score_negative_words(fp_twt.text)
#print fp_twt.pivot_table(values='negative', index=['src','article'],aggfunc=np.sum)
#print fp_twt.pivot_table(values='negative', index=['src','article'],aggfunc=np.mean)
#print fp_twt.pivot_table(values='negative', index=['src','article'],aggfunc=len)

    #    for i,tweet in enumerate(all_tweets['content']['statuses']):
    #        print( '%d %s @%s tweeted: %s' % (i, tweet['created_at'], tweet['user']['screen_name'], tweet['text'] ) )
    #    print( "Current rate-limiting status: %s" % ts.get_metadata()['x-rate-limit-reset'])
    # check all tweets according to their ID

