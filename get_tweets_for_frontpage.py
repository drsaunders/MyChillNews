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
import psycopg2
import re
import urllib
import base64
import cStringIO
import pycurl
import json
from sqlalchemy import create_engine
import sqlalchemy


#%%

    
def clear_tweets_for_day(engine, timestamp):
    con = engine.connect()

    rs = con.execute("UPDATE frontpage SET all_tweets_collected=FALSE WHERE fp_timestamp='%s';" % timestamp)

    rs = con.execute("DELETE FROM tweets WHERE fp_timestamp='%s';" % timestamp)
    con.close()
    #%%
    
def check_day(engine, timestamp):
    sql_query = "SELECT src, headline, article_order, all_tweets_collected FROM frontpage WHERE fp_timestamp='%s' AND article_order<=10 ORDER BY src, article_order;" % timestamp
    to_check = pd.read_sql_query(sql_query,engine)
    to_check.columns =['a','b','c','d']
    prev = pd.options.display.max_rows
    pd.options.display.max_rows = 999
    pd.options.display.width = 1
    pd.options.display.max_colwidth = 50
    print(to_check)
    pd.options.display.max_rows = prev
    
    #%%
def get_reset_info(ts):
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords(['cheese'])
    tso.set_include_entities(False) # and don't give us all those entity information
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
        in15 = datetime.now() + timedelta(minutes=15)
        print "Should reset by: " + str(in15.time())
        raise

    if len(all_tweets) > 0:
        oldest_created_date = dateutil.parser.parse(all_tweets[-1]['created_at'])
        oldest_created_date = oldest_created_date.astimezone(timezone('US/Eastern'))
        print "Total retrieved: %d" % len(all_tweets)
        print "Oldest tweet retrieved was created " + str(oldest_created_date)
      
    print "Time taken for twitter search: %.1f seconds" % ((time.time() - start_time)/1.)
    print "Number of searches left this time slice: " + response['meta']['x-rate-limit-remaining']
    return all_tweets

def get_all_tweets_for_article(article, engine, ts):
 
    url = article.url
    if article.src == 'lat':
        search_url = re.sub('http://','http:/',url)
    else:
        search_url = url
        
    article_tweet_list = []
    try:
        tweets_for_article = get_all_tweets_for_search(search_url, ts)
    except TwitterSearchException as e:
        raise
        
    twitter_search_timestamp =  datetime.now().strftime("%Y-%m-%d-%H%M")
    for j,tweet in enumerate(tweets_for_article):
        article_tweet_list.append({'src':article.src,                                         
        'fp_timestamp':article.fp_timestamp,
        'article':article.article_order,
        'order':j,  
        'id':str(tweet['id']),
        'created':tweet['created_at'],
        'user':tweet['user']['screen_name'],
        'text':tweet['text'],
        'retweet_count':tweet['retweet_count'],
        'retrieved_at':twitter_search_timestamp
            })
    
    tweet_download_record = pd.DataFrame({'retrieved_at':twitter_search_timestamp, 'url':url, 'src':article.src, 'article':article.article_order}, index=[0])  # 'how_many':len(tweets_for_article)
   
    # Overwrite the current frontpage table
    tweet_download_record.to_sql('tweet_download_log', engine, if_exists='append')
    
    return article_tweet_list



#%%
# MAIN 
 
with open('../twitter_app_keys.dat','r') as f:
    consumer_key = next(f).strip()
    consumer_secret = next(f).strip()
    access_token = next(f).strip()
    access_token_secret = next(f).strip()

try:
    ts = TwitterSearch(consumer_key = consumer_key, consumer_secret = consumer_secret,
        access_token = access_token,
        access_token_secret = access_token_secret
    )
except TwitterSearchException as e:
    print "TWITTER ERROR: Too many simultaneous connections"
    print e

#%%

if __name__ == "__main__":
    
    total_start_time = time.time()
    total_num_searches = 0
    fp_timestamp = '2016-09-21-0842'
    
    dbname = 'frontpage'
    username = 'dsaunder'
    
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    #pcon = None
    #con = psycopg2.connect(database = dbname, user = username)
    
    # Select all the headlines for this timestamp that did not already have their 
    # tweets downloaded
    sql_query = '''
        SELECT * 
        FROM frontpage 
        WHERE fp_timestamp='%s' 
        AND article_order <= 10
        AND url NOT IN (
           SELECT frontpage.url 
           FROM frontpage 
               JOIN tweet_download_log ON frontpage.url = tweet_download_log.url);
        '''  % fp_timestamp
    
        
    frontpage_data = pd.read_sql_query(sql_query,engine,index_col='index')
    #%%
    
    #frontpage_data = pd.read_csv(timestamp + '_frontpage_data.csv')
    new_tweet_list = []
    try:
        for src in np.unique(frontpage_data.src):
            articles_for_paper = frontpage_data.loc[frontpage_data.src ==src, :]
            for i in range(0,len(articles_for_paper)):
                new_tweet_list.extend(get_all_tweets_for_article(articles_for_paper.iloc[i,:], engine, ts))
    
    except TwitterSearchException as e:
        print "Did not complete source %s" % src 
                    #%%
    new_tweets = pd.DataFrame(new_tweet_list)
    tweets_retrieved = len(new_tweets)

    #%%
    use_row = []
    if len(new_tweets) > 0:
        
        sql_query = "SELECT id FROM tweets WHERE fp_timestamp = '%s';" % fp_timestamp
        
        existing_ids = pd.read_sql_query(sql_query,engine)
        existing_ids = set(existing_ids.values.flat)
        use_row = np.invert(new_tweets.id.isin(existing_ids))
        if np.sum(use_row) > 0:
            # Add the new tweets to the collection
            new_tweets.loc[use_row,:].to_sql('tweets', engine, if_exists='append')
            
    print "Total time elapsed: %.1f minutes." % ((time.time() - total_start_time ) / 60.)
    print "Total tweets retrieved: %d" % tweets_retrieved
    print("Wrote %d new tweets to database" % np.sum(use_row))
