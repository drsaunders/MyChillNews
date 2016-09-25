#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:32:43 2016

@author: dsaunder
"""


import httplib
import urlparse
import re
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import scipy

dbname = 'frontpage'
username = 'dsaunder'
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
#
#sql_query = """
#        SELECT * 
#        FROM fb_statuses 
#     """

sql_query = """
        SELECT * 
        FROM fb_statuses 
        WHERE status_id NOT IN
            (SELECT status_id FROM fb_matchstr_lookup);
    """

fb = pd.read_sql_query(sql_query,engine)

def unshorten_url(url):
    parsed = urlparse.urlparse(url)
    h = httplib.HTTPConnection(parsed.netloc)
    resource = parsed.path
    if parsed.query != "":
        resource += "?" + parsed.query
    h.request('HEAD', resource )
    response = h.getresponse()
    if response.status/100 == 3 and response.getheader('Location'):
        return unshorten_url(response.getheader('Location')) # changed to process chains of short urls
    else:
        return url
        
        

matchstr_lookup = []
for i,row in fb.iterrows():
    if row.src == 'nyt':
        if not 'nyti.ms' in row.status_link:
            continue
        url = unshorten_url(row.status_link)
        if not 'myaccount' in url:
            continue
        pattern = '%2F([^%.]*\.html)'
    elif row.src == 'lat':
        url = row.status_link
        pattern = '(.*)'
    elif row.src == 'cnn':
        if not 'cnn.it' in row.status_link:
            continue
        url = unshorten_url(row.status_link)
        pattern = '/([^/]*)/index\.html'
    elif row.src == 'dm':
        if not 'dailym.ai' in row.status_link:
            continue
        url = unshorten_url(row.status_link)
        pattern = '/([^/]*\.html)'
    elif row.src == 'fox':
        if not '.foxnews.com' in row.status_link:
            continue
        if 'latino.foxnews.com' in row.status_link:
            continue
        url = row.status_link
        pattern = '/([^/]*)($|\.html)'
    elif row.src == 'bbc':
        if not 'bbc.in' in row.status_link:
            continue
        url = unshorten_url(row.status_link)
        pattern = '/([^/]*)\?'
    elif row.src == 'usa':
        if not '.usatoday.com' in row.status_link:
            continue
        url = row.status_link
        pattern = '/([^/]*/[^/]*)/[^/]*/$'
    elif row.src == 'wsj':
        url = row.status_link
        if 'on.wsj.com' in url:
            url = unshorten_url(url)
        pattern = '/([^/]+)\?'
    elif row.src == 'gua':
        url = row.status_link
        pattern = '/([^/]+)\?'
    elif row.src == 'wap':
        url = row.status_link
#        print url
        if 'wapo.st' in url:
            r = requests.head(url, allow_redirects=True)
            url = r.url
        if not 'www.washingtonpost.com' in url:
            continue
        
        pattern = '/([^/]+)(\.html|/\?)'
    else:
        continue
    
    thematch = re.search(pattern,url)
    if thematch:
        matchstr = re.search(pattern,url).groups()[0]
#    print url + "\n     " + matchstr
        if len(matchstr) > 0:
            print row.src + ":   " + matchstr            
            matchstr_lookup.append({'src':row.src, 'status_id':row.status_id, 'when_posted':row.status_published, 'url':url, 'matchstr':matchstr})

matchstr_lookup = pd.DataFrame(matchstr_lookup)
#matchstr_lookup.to_sql('fb_matchstr_lookup', engine, if_exists='replace')
matchstr_lookup.to_sql('fb_matchstr_lookup', engine, if_exists='append')
print "%d fb matchstrings added" % len(matchstr_lookup)