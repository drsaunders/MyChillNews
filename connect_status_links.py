#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:31:58 2016

@author: dsaunder
"""

import httplib
import urlparse
import re
import requests
import pandas as pd


sql_query = "SELECT * FROM fb_statuses;"
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
        
        

mappings = []
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
        print row.src + ":   " + matchstr            

        mappings.append({'src':row.src, 'status_id':row.status_id, 'when_posted':row.status_published, 'url':url, 'matchstr':matchstr})

mappings = pd.DataFrame(mappings)
mappings.to_sql('fb_status_mapping', engine, if_exists='replace')
#%%

sql_query = "SELECT * FROM frontpage;" # WHERE article_order <= 10;"
frontpage_data = pd.read_sql_query(sql_query,engine)
#%%
nummatches = 0
for i,row in frontpage_data.iterrows():
     matches = [a in row.url for a in mappings.loc[mappings.src==row.src,'matchstr']]
     if (np.sum(matches) >= 1):
         print row.src + "   " + str(np.sum(matches)) + " " + row.headline
         for mappingindex in np.nonzero(matches)[0]:
             the_mapping = mappings.loc[mappings.src==row.src].iloc[mappingindex]

         nummatches = nummatches + 1
             #             print "   " + the_mapping.matchstr + "   " + the_mapping.url
              #%%        
print nummatches         

#%%
for i in range(len(fb)):
    if ('skittles' in fb.iloc[i].status_link) and ('fox' in fb.iloc[i].src):
        print fb.iloc[i].status_link



#%%
