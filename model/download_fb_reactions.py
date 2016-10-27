#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download a large batch of recent FB news stories and add them to the database.

For all my news sources, go to their public facebook pages and scrape the most
recent 200 stories worth of reactions and add them to the database (removing
duplicates first)

This all happens on my local computer, where I train the model, rather than 
on the app.

Quite slow at the moment - the status processing could almost certainly be 
optimized.

Most of the code comes from this original source.
# Forked from https://github.com/minimaxir/facebook-page-post-scraper, originally by Max Woolf (minimaxir), https://github.com/minimaxir
# Updated by Peeter Tinits 14.06.2016 to read reactions separately based on http://stackoverflow.com/questions/36930414/how-can-i-get-facebook-graph-api-reaction-summary-count-separately recommendations.
# Runs with Py 2.7.10

"""

import urllib2
import json
import datetime
import csv
import time
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import numpy as np
import getpass
#%%

# Connect to the database
dbname = 'frontpage'
username = getpass.getuser()
if username == 'root':  # Hack just for my web server
    username = 'ubuntu'
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

#Get your app id and key here on facebook. Help here: https://goldplugins.com/documentation/wp-social-pro-documentation/how-to-get-an-app-id-and-secret-key-from-facebook/

with open('../../facebook_app_keys.dat','r') as f:
    app_id = next(f).strip()
    app_secret = next(f).strip()

access_token = app_id + "|" + app_secret

limited = True # limits the number of posts processed. Change to False if you want the whole page.
limit = 200 # The number of pages to be processed if limited is True. Uses multiples of 100 to scrape for bandwith reasons.

def request_until_succeed(url):
    """Repeat request for URL until it works.
    
    Args:
        url: The url to open
       
    Return:
        The contents of the page
    """
    req = urllib2.Request(url)
    success = False
    while success is False:
        try: 
            response = urllib2.urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception, e:
            print e
            time.sleep(5)
            
            print "Error for URL %s: %s" % (url, datetime.datetime.now())

    return response.read()


def unicode_normalize(text):
    """ Needed to write tricky unicode correctly to csv
    """
    return text.translate({ 0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22, 0xa0:0x20 }).encode('utf-8')

def getFacebookPageFeedData(page_id, access_token, num_statuses):
    """Gets a list of latest statuses from a particular page.
    
    Args:
        page_id: The FB page to download from.
        access_token: FB developer api token 
        num_statuses: How many to grab this time
        
    """
    # construct the URL string
    base = "https://graph.facebook.com/v2.6/"
    node = "/" + page_id + "/posts" 
    parameters = "/?fields=message,link,created_time,type,name,id,comments.limit(1).summary(true),shares,reactions.limit(1).summary(true)&limit=%s&access_token=%s" % (num_statuses, access_token) # changed
    url = base + node + parameters
    
    # retrieve data
    data = json.loads(request_until_succeed(url))
    
    return data
    
def getFacebookPostData(page_id, access_token):
    """Gets the whole JSON of info about a particular post.
    
    Args:
        page_id: The id of the particular post
        access_token: FB developer api token 
         
    Returns:
        The whole JSON of stuff about the post.
    """
    num_statuses = 1
    # construct the URL string
    base = "https://graph.facebook.com/v2.6/"
    node = page_id 
    #parameters = "/?fields=reactions.summary(true),reactions.type(LIKE).limit(0).summary(true).as(like),reactions.type(LOVE).limit(0).summary(true).as(love),reactions.type(WOW).limit(0).summary(true).as(wow),reactions.type(HAHA).limit(0).summary(true).as(haha),reactions.type(SAD).limit(0).summary(true).as(sad),reactions.type(ANGRY).limit(0).summary(true).as(angry),reactions.type(THANKFUL).limit(0).summary(true).as(thankful)&limit=%s&access_token=%s" % (num_statuses, access_token) # changed

    #Potentially quicker, but maybe no difference:
    parameters = "/?fields=reactions.summary(total_count),reactions.type(LIKE).limit(0).summary(total_count).as(like),reactions.type(LOVE).limit(0).summary(total_count).as(love),reactions.type(WOW).limit(0).summary(total_count).as(wow),reactions.type(HAHA).limit(0).summary(total_count).as(haha),reactions.type(SAD).limit(0).summary(total_count).as(sad),reactions.type(ANGRY).limit(0).summary(total_count).as(angry),reactions.type(THANKFUL).limit(0).summary(total_count).as(thankful)&limit=%s&access_token=%s" % (num_statuses, access_token) # changed
    url = base + node + parameters
    
    # retrieve data
    data = json.loads(request_until_succeed(url))
    #print(data) - for debugging
    return data
    
   

def processFacebookPageFeedStatus(status):
    """Takes a single FB story and break it down into a row.

    Args:
        status: A facebook story retrieved from an API call

    Returns:
        A single row describing the aggregated reactions to the story.    
    """
    # The status is now a Python dictionary, so for top-level items,
    # we can simply call the key.
    
    # Additionally, some items may not always exist,
    # so must check for existence first
    #print(status['id'])
    status_id = status['id']

    # To access data on particular reactions, accessing by post was easier than accessing by page.
    # data is returned with each reaction data point.
    # This slows down the script a bit and probably better ways are out there.
    data = getFacebookPostData(status_id,access_token)

    
    status_message = '' if 'message' not in status.keys() else unicode_normalize(status['message'])
    link_name = '' if 'name' not in status.keys() else unicode_normalize(status['name'])
    status_type = status['type']
    status_link = '' if 'link' not in status.keys() else unicode_normalize(status['link'])
    
    
    # Time needs special care since a) it's in UTC and
    # b) it's not easy to use in statistical programs.
    
    status_published = datetime.datetime.strptime(status['created_time'],'%Y-%m-%dT%H:%M:%S+0000')
    status_published = status_published + datetime.timedelta(hours=-5) # EST
    status_published = status_published.strftime('%Y-%m-%d %H:%M:%S') # best time format for spreadsheet programs
    
    # Nested items require chaining dictionary keys.
    
    #num_reactions1 = 0 if 'reactions' not in status.keys() else status['reactions']['type']['like']#.limit(0).summary(true).as(like)
    num_reactions = 0 if 'reactions' not in status.keys() else status['reactions']['summary']['total_count']
    num_comments = 0 if 'comments' not in status.keys() else status['comments']['summary']['total_count']
    num_shares = 0 if 'shares' not in status.keys() else status['shares']['count']
    num_likes = 0 if 'like' not in data.keys() else data['like']['summary']['total_count']
    num_loves = 0 if 'love' not in data.keys() else data['love']['summary']['total_count']
    num_wows = 0 if 'wow' not in data.keys() else data['wow']['summary']['total_count']
    num_hahas = 0 if 'haha' not in data.keys() else data['haha']['summary']['total_count']
    num_sads = 0 if 'sad' not in data.keys() else data['sad']['summary']['total_count']
    num_angries = 0 if 'angry' not in data.keys() else data['angry']['summary']['total_count']
    num_thankfuls = 0 if 'thankful' not in data.keys() else data['thankful']['summary']['total_count']
    # return a tuple of all processed data
    return (status_id, status_message, link_name, status_type, status_link,
           status_published, num_reactions, num_comments, num_shares, num_likes,
            num_loves, num_wows, num_hahas, num_sads, num_angries, num_thankfuls)

def scrapeFacebookPageFeedStatus(page_id, access_token):
    """Gets all the statuses, up to a particular limit, for a public FB page.
    
    Args:
        page_id: The name of the public page e.g. "HuffingtonPost"
        access_token: FB developer api token 
        
    Returns:
        A dataframe with a row for each story, and columns as described below.
    """
    columns=["status_id", "status_message", "link_name", "status_type", "status_link",
       "status_published", "num_reactions", "num_comments", "num_shares", "num_likes",
                "num_loves","num_wows","num_hahas","num_sads","num_angries","num_thankfuls"]
    
    has_next_page = True
    num_processed = 0   # keep a count on how many we've processed
    scrape_starttime = datetime.datetime.now()
    
    print "Scraping %s Facebook Page: %s\n" % (page_id, scrape_starttime)
    
    statuses = getFacebookPageFeedData(page_id, access_token, 100)
    #print(statuses)
    
    # Get one "page" of 100 statuses at a time
    new_rows = []
    while has_next_page:
        for status in statuses['data']:
            status_items = processFacebookPageFeedStatus(status)
            
            new_dict = dict()
            for i,c in enumerate(columns):
                new_dict[c] = status_items[i]
            new_rows.append(new_dict)    
                
            # output progress occasionally to make sure code is not stalling
            num_processed += 1
            if num_processed % 20 == 0:
                print "%s Statuses Processed: %s" % (num_processed, datetime.datetime.now())
            if limited == True and num_processed >= limit:
                has_next_page = False
                
        # if there is no next page, we're done.
        if 'paging' in statuses.keys():
            statuses = json.loads(request_until_succeed(statuses['paging']['next']))
        else:
            has_next_page = False
     
    
    print "\nDone!\n%s Statuses Processed in %s" % (num_processed, datetime.datetime.now() - scrape_starttime)

    new_statuses = pd.DataFrame(new_rows)
    return new_statuses
    
if __name__ == '__main__':
    dbname = 'frontpage'
    username = getpass.getuser()
    if username == 'root':  # Hack just for my web server
        username = 'ubuntu'

    timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
   
    # Get the news source info
    sql_query = "SELECT * FROM srcs;"
    srcs = pd.read_sql_query(sql_query,engine,index_col='index')

    # Query the existing linked URLs to avoid adding duplicates
    sql_query = "SELECT status_link FROM fb_statuses;"
    existing_ids = pd.read_sql_query(sql_query,engine)
    existing_ids = set(existing_ids.values.flat)

    # Go through all the sources and download the last 200 FB statuses for the 
    # associated public pages
    total_written = 0
    for i,src_row in srcs.iterrows():
        if src_row.fb_page == '':
            continue
        src_statuses = scrapeFacebookPageFeedStatus(src_row.fb_page, access_token)
        src_statuses.loc[:,'src'] = src_row.prefix
        src_statuses.loc[:,'retrieved_time'] = timestamp

        use_row = np.invert(src_statuses.status_link.isin(existing_ids))

        src_statuses.loc[use_row,:].to_sql('fb_statuses', engine, if_exists='append')

        total_written = total_written + np.sum(use_row)
        print ("Earliest for %s is " % src_row.prefix)
        print src_statuses.status_published.sort_values().iloc[0]
    print("Wrote %d new headlines to database" % total_written)
