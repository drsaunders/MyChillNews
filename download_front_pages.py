#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:36:36 2016

@author: dsaunder
"""

import requests
import os
import datetime

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
        'front_page':'http://washingtonpost.com/'},
        {'name':'Google News', 
         'prefix':'goo',
        'front_page':'http://news.google.com'},
        {'name':'Huffington Post', 
         'prefix':'huf',
        'front_page':'http://www.huffingtonpost.com/'},
       {'name':'CNN', 
         'prefix':'cnn',
        'front_page':'http://www.cnn.com/'},
       {'name':'NBC news', 
         'prefix':'nbc',
        'front_page':'http://www.nbcnews.com/'},
        {'name':'Daily Mail', 
         'prefix':'dm',
        'front_page':'http://www.dailymail.co.uk/home/index.html'},
        {'name':'ABC News', 
         'prefix':'abc',
        'front_page':'http://abcnews.go.com/'},
        {'name':'Wall Street Journal', 
         'prefix':'wsj',
        'front_page':'http://www.wsj.com/'},
        {'name':'BBC News', 
         'prefix':'bbc',
        'front_page':'http://www.bbc.com/news'},
        {'name':'USA Today', 
         'prefix':'usa',
        'front_page':'http://www.usatoday.com/'},
        {'name':'The Guardian', 
         'prefix':'gua',
        'front_page':'https://www.theguardian.com/uk?INTCMP=CE_UK'}
        ]

        
        #%%
timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
frontpagedir = 'frontpages/%s/' % timestamp
if not os.path.exists(frontpagedir):
    os.makedirs(frontpagedir)

for src in srcs:
    response = requests.get(src['front_page'])
    if response.status_code == 200:    
        outfile = frontpagedir + src['prefix'] + timestamp + '.html'
        with open(outfile, 'w') as f:
            f.write(response.content)
    else:
        print "Failed to access URL %s" % src['front_page']
#%%

