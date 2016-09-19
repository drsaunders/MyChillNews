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

total_start_time = time.time()
total_num_searches = 0
timestamp = '2016-09-19-0722'


# obtains the bearer token
def get_bearer_token(consumer_key,consumer_secret):
    # enconde consumer key
    consumer_key = urllib.quote(consumer_key)
    # encode consumer secret
    consumer_secret = urllib.quote(consumer_secret)
    # create bearer token
    bearer_token = consumer_key+':'+consumer_secret
    # base64 encode the token
    base64_encoded_bearer_token = base64.b64encode(bearer_token)
    # set headers
    headers = [
    "POST /oauth2/token HTTP/1.1", 
    "Host: api.twitter.com", 
    "User-Agent: jonhurlock Twitter Application-only OAuth App Python v.1",
    "Authorization: Basic "+base64_encoded_bearer_token+"",
    "Content-Type: application/x-www-form-urlencoded;charset=UTF-8", 
    "Content-Length: 29"]
    # do the curl
    token_url = "https://api.twitter.com/oauth2/token"
    buf = cStringIO.StringIO()
    access_token = ''
    pycurl_connect = pycurl.Curl()
    pycurl_connect.setopt(pycurl_connect.URL, token_url) # used to tell which url to go to
    pycurl_connect.setopt(pycurl_connect.WRITEFUNCTION, buf.write) # used for generating output
    pycurl_connect.setopt(pycurl_connect.HTTPHEADER, headers) # sends the customer headers above
    pycurl_connect.setopt(pycurl_connect.POSTFIELDS, 'grant_type=client_credentials') # sends the post data
    #pycurl_connect.setopt(pycurl_connect.VERBOSE, True) # used for debugging, really helpful if you want to see what happens
    pycurl_connect.perform() # perform the curl
    access_token = buf.getvalue() # grab the data
    pycurl_connect.close() # stop the curl
    x = json.loads(access_token)
    bearer = x['access_token']
    return bearer

def search_tweets(bearer_token, query):
    # url to perform search
    url = "https://api.twitter.com/1.1/search/tweets.json"
    url_params = '?q='+query
    url_params = url_params +'&include_entities=false'
    url_params = url_params +'&lang=en'
#    url_params = url_params + ' -filter:retweets'
#    
    headers = [ 
    str("GET /1.1/search/tweets.json"+url_params+" HTTP/1.1"), 
    str("Host: api.twitter.com"), 
    str("User-Agent: jonhurlock Twitter Application-only OAuth App Python v.1"),
    str("Authorization: Bearer "+bearer_token+"")
    ]
    buf = cStringIO.StringIO()
    results = ''
    pycurl_connect = pycurl.Curl()
    retrieved_headers = Storage()

    print url+url_params
    pycurl_connect.setopt(pycurl_connect.URL, url+url_params) # used to tell which url to go to
    pycurl_connect.setopt(pycurl_connect.WRITEFUNCTION, buf.write) # used for generating output
    pycurl_connect.setopt(pycurl_connect.HTTPHEADER, headers) # sends the customer headers above
    pycurl_connect.setopt(pycurl_connect.HEADERFUNCTION, retrieved_headers.store)
    pycurl_connect.setopt(pycurl_connect.VERBOSE, True) # used for debugging, really helpful if you want to see what happens
    pycurl_connect.perform() # perform the curl
    results += buf.getvalue() # grab the data
    pings_left = grab_rate_limit_remaining(retrieved_headers)
    pycurl_connect.close() # stop the curl
    print '_pings_left %s' % pings_left

    return json.loads(results)

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
 
with open('../twitter_app_keys.dat','r') as f:
    consumer_key = next(f).strip()
    consumer_secret = next(f).strip()

# obtains the bearer token
bearer_token = get_bearer_token(consumer_key,consumer_secret)

#%%
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
    WHERE url NOT IN (
       SELECT frontpage.url 
       FROM frontpage 
           JOIN tweet_download_log ON frontpage.url = tweet_download_log.url)
     AND fp_timestamp='%s';
    '''  % timestamp

    
frontpage_data = pd.read_sql_query(sql_query,engine,index_col='index')
#%%

tweets_retrieved = 0
#frontpage_data = pd.read_csv(timestamp + '_frontpage_data.csv')
new_tweet_list = []
try:
    for src in np.unique(frontpage_data.src):
        articles_for_paper = frontpage_data.loc[(frontpage_data.src ==src) & (frontpage_data.article_order <=10),: ]
        for i in range(0,len(articles_for_paper)):

            url = articles_for_paper.iloc[i,:].url
            if src == 'lat':
                search_url = re.sub('http://','http:/',url)
            else:
                search_url = url
                
            tweets_for_article = get_all_tweets_for_search(search_url, ts)
            twitter_search_timestamp =  datetime.now().strftime("%Y-%m-%d-%H%M")
            tweets_retrieved = tweets_retrieved + len(tweets_for_article)
            for j,tweet in enumerate(tweets_for_article):
                new_tweet_list.append({'src':src,                                         
                'fp_timestamp':timestamp,
                'article':i+1,
                'order':j,  
                'id':str(tweet['id']),
                'created':tweet['created_at'],
                'user':tweet['user']['screen_name'],
                'text':tweet['text'],
                'retweet_count':tweet['retweet_count'],
                'retrieved_at':twitter_search_timestamp
                    })
            
            tweet_download_record = pd.DataFrame({'retrieved_at':twitter_search_timestamp, 'url':url, 'src':src, 'article':i+1}, index=[0])
           
            # Overwrite the current frontpage table
            tweet_download_record.to_sql('tweet_download_log', engine, if_exists='append')


except TwitterSearchException as e:
    print "Did not complete source %s" % src 
                #%%
new_tweets = pd.DataFrame(new_tweet_list)

#%%
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
