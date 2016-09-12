#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:20:37 2016

@author: dsaunder
"""

from bs4 import BeautifulSoup
import glob
import pandas as pd
import re

srcs = [
        {'name':'New York Times', 
         'prefix':'nyt',
        'front_page':'http://www.nytimes.com/pages/todayspaper/index.html'},
        
        {'name':'Yahoo News', 
         'prefix':'yahoo',
        'front_page':'https://www.yahoo.com/news/?ref=gs'},
        
        {'name':'LA Times', 
         'prefix':'lat',
        'front_page':'http://latimes.com/'},

        {'name':'Fox News', 
         'prefix':'fox',
        'front_page':'http://foxnews.com'},
        {'name':'Washington Post', 
         'prefix':'wap',
        'front_page':'http://washingtonpost.com/'}
        ]
#%%

def read_frontpage_by_prefix(prefix, frontpagedir):
    return glob.glob(frontpagedir + prefix + '*')[0]
    
def get_url(tag, url_prefix=None):
    url = tag.attrs['href']
    url = url.split('#')[0]
    url = url.split('?')[0]
    if url_prefix:
        if not 'http' in url:
            url = url_prefix + url
    return url
    
def get_contents(tag):
    contents = tag.decode_contents()
    if not contents:
        return ''
        
    if type(contents) == list:
        contents = contents[0]

    contents = contents.strip()

    contents, dummy = re.subn('&amp;apos;','\'',contents)
    return contents
    #%%
    #timestamp = '2016-09-08-1518'
timestamp = '2016-09-12-0717'
frontpagedir = 'frontpages/%s/' % timestamp

    #%%
frontpage_data = pd.DataFrame()
#%%
# LA TIMES
prefix = 'lat'
url_prefix = 'http://latimes.com'
with open(read_frontpage_by_prefix(prefix,frontpagedir), 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
headline_selectors = ['a.trb_outfit_primaryItem_article_title_a','a.trb_outfit_relatedListTitle_a']

src_rows = pd.DataFrame()
for selector in headline_selectors:
    headlines = soup.select(selector)
    
    new_rows = pd.DataFrame({'src':[prefix]*len(headlines), 
     'headline':[get_contents(a) for a in headlines],
     'url':[get_url(a, url_prefix) for a in headlines],
    })
    
    new_rows = new_rows.loc[new_rows.headline != '', :]
    src_rows = src_rows.append(new_rows, ignore_index=True)

src_rows.loc[:,'article_order'] = range(1,len(src_rows)+1)
frontpage_data = frontpage_data.append(src_rows, ignore_index=True)

#%%
# New York Times
prefix = 'nyt'
url_prefix = None

with open(read_frontpage_by_prefix(prefix,frontpagedir), 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
headline_selectors = ['div.story h3 a','ul.headlinesOnly h6 a']

src_rows = pd.DataFrame()
for selector in headline_selectors:
    headlines = soup.select(selector)
    
    new_rows = pd.DataFrame({'src':[prefix]*len(headlines), 
     'headline':[get_contents(a) for a in headlines],
     'url':[get_url(a, url_prefix) for a in headlines]
    })
    
    new_rows = new_rows.loc[new_rows.headline != '', :]
    src_rows = src_rows.append(new_rows, ignore_index=True)

src_rows.loc[:,'article_order'] = range(1,len(src_rows)+1)
frontpage_data = frontpage_data.append(src_rows, ignore_index=True)
    

##%%
## Google News
#prefix = 'goo'
#url_prefix = None
#
#with open(read_frontpage_by_prefix(prefix,frontpagedir), 'r') as f:
#    soup = BeautifulSoup(f, 'html.parser')
#headline_selectors = ['.esc-lead-article-title']
#
#src_rows = pd.DataFrame()
#for selector in headline_selectors:
#    headlines = soup.select(selector)
#    headlines = [a.contents[0] for a in headlines]
#    new_rows = pd.DataFrame({'src':[prefix]*len(headlines), 
#     'headline':[get_contents(a.contents[0]) for a in headlines],
#     'url':[get_url(a, url_prefix) for a in headlines]
#    })
#    
#    new_rows = new_rows.loc[new_rows.headline != '', :]
#    src_rows = src_rows.append(new_rows, ignore_index=True)
#
#src_rows.loc[:,'article_order'] = range(1,len(src_rows)+1)
#frontpage_data = frontpage_data.append(src_rows, ignore_index=True)
#    

#%%
# CNN
prefix = 'cnn'
url_prefix = 'http://www.cnn.com'

with open(read_frontpage_by_prefix(prefix,frontpagedir), 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
headline_selectors = ['h3.cd__headline']

src_rows = pd.DataFrame()
for selector in headline_selectors:
    headlines = soup.select(selector)
    headlines = [a.contents[0] for a in headlines]
    new_rows = pd.DataFrame({'src':[prefix]*len(headlines), 
     'headline':[a.contents[0].contents[0] for a in headlines],
     'url':[get_url(a, url_prefix) for a in headlines]
    })
    
    new_rows = new_rows.loc[new_rows.headline != '', :]
    src_rows = src_rows.append(new_rows, ignore_index=True)

src_rows.loc[:,'article_order'] = range(1,len(src_rows)+1)
frontpage_data = frontpage_data.append(src_rows, ignore_index=True)

#%%
# Fox
prefix = 'fox'
url_prefix = None

with open(read_frontpage_by_prefix(prefix,frontpagedir), 'r') as f:
    soup = BeautifulSoup(f, 'lxml')
headline_selectors = ['.primary h1 a','.top-stories a']

src_rows = pd.DataFrame()
for i,selector in enumerate(headline_selectors):
    headlines = soup.select(selector)
    new_rows = pd.DataFrame({'src':[prefix]*len(headlines), 
     'headline':[get_contents(a) for a in headlines],
     'url':[get_url(a, url_prefix) for a in headlines]
    })
    
    new_rows = new_rows.loc[new_rows.headline != '', :]
    src_rows = src_rows.append(new_rows, ignore_index=True)

for j in range(len(src_rows)):
    if '<!--' in src_rows.loc[j,'headline']:
        src_rows.loc[j,'headline'] = re.search('<h3>(.*)</h3>',src_rows.loc[j,'headline']).groups()[0]
    if 'span style' in src_rows.loc[j,'headline']:
        src_rows.loc[j,'headline'] = re.search('>(.*)<',src_rows.loc[j,'headline']).groups()[0]

src_rows.loc[:,'article_order'] = range(1,len(src_rows)+1)
frontpage_data = frontpage_data.append(src_rows, ignore_index=True)





#%%
frontpage_data.to_csv(timestamp + '_frontpage_data.csv',index=False, encoding='utf-8')
