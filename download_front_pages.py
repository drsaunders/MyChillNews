#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:36:36 2016

@author: dsaunder
"""

import requests
import os
import datetime
import pandas as pd
from sqlalchemy import create_engine
from subprocess import Popen, PIPE
from selenium import webdriver


from subprocess import Popen, PIPE
from selenium import webdriver
from sqlalchemy import create_engine

abspath = lambda *p: os.path.abspath(os.path.join(*p))
ROOT = abspath(os.path.dirname(__file__))
#%%
def create_srcs_table(engine):
    srcs = [
            {'name':'New York Times', 
            'prefix':'nyt',
            'fb_page':'nytimes',
            'front_page':'http://www.nytimes.com/pages/todayspaper/index.html'},
            {'name':'Yahoo News',
            'prefix':'yahoo',
            'fb_page':'yahoonews',
            'front_page':'https://www.yahoo.com/news/?ref=gs'},
            
            {'name':'LA Times', 
             'prefix':'lat',
            'fb_page':'latimes',
             'front_page':'http://latimes.com/'},
 
            {'name':'Fox News', 
             'prefix':'fox',
            'fb_page':'FoxNews',
            'front_page':'http://foxnews.com'},
            {'name':'Washington Post', 
             'prefix':'wap',
            'fb_page':'washingtonpost',
            'front_page':'http://washingtonpost.com/'},
            {'name':'Google News', 
             'prefix':'goo',
            'fb_page':'',
            'front_page':'http://news.google.com'},
            {'name':'Huffington Post', 
             'prefix':'huf',
            'fb_page':'HuffingtonPost',
            'front_page':'http://www.huffingtonpost.com/'},
            {'name':'CNN', 
             'prefix':'cnn',
            'fb_page':'cnn',
            'front_page':'http://www.cnn.com/'},
            {'name':'NBC news', 
             'prefix':'nbc',
            'fb_page':'NBCNews',
            'front_page':'http://www.nbcnews.com/'},
            {'name':'Daily Mail', 
             'prefix':'dm',
            'fb_page':'DailyMail',
            'front_page':'http://www.dailymail.co.uk/home/index.html'},
            {'name':'ABC News', 
             'prefix':'abc',
            'fb_page':'abcnews',
            'front_page':'http://abcnews.go.com/'},
            {'name':'Wall Street Journal', 
             'prefix':'wsj',
            'fb_page':'wsj',
            'front_page':'http://www.wsj.com/'},
            {'name':'BBC News', 
             'prefix':'bbc',
            'fb_page':'bbcnews',
            'front_page':'http://www.bbc.com/news'},
            {'name':'USA Today', 
             'prefix':'usa',
            'fb_page':'usatoday',
            'front_page':'http://www.usatoday.com/'},
            {'name':'The Guardian', 
             'prefix':'gua',
            'fb_page':'theguardian',
            'front_page':'https://www.theguardian.com/uk?INTCMP=CE_UK'}]
    srcs = pd.DataFrame(srcs)
    srcs.to_sql('srcs', engine, if_exists='replace')

#%%
def execute_command(command):
    result = Popen(command, shell=True, stdout=PIPE).stdout.read()
    if len(result) > 0 and not result.isspace():
        raise Exception(result)


def do_screen_capturing(url, screen_path, width, height):
    print "Capturing screen.."
    driver = webdriver.PhantomJS()
    # it save service log file in same directory
    # if you want to have log file stored else where
    # initialize the webdriver.PhantomJS() as
    # driver = webdriver.PhantomJS(service_log_path='/var/log/phantomjs/ghostdriver.log')
    driver.set_script_timeout(30)
    if width and height:
        driver.set_window_size(width, height)
    driver.get(url)
    driver.save_screenshot(screen_path)


def do_crop(params):
    command = [
        'convert',
        params['screen_path'],
        '-crop', '%sx%s+0+0' % (params['width'], params['height']),
        params['crop_path']
    ]
    print command
    execute_command(' '.join(command))


def do_thumbnail(params):
    print "Generating thumbnail from croped captured image.."
    command = [
        'convert',
        params['crop_path'],
        '-filter', 'Lanczos',
        '-thumbnail', '%sx%s' % (params['width'], params['height']),
        params['thumbnail_path']
    ]
    execute_command(' '.join(command))


def get_screen_shot(**kwargs):
    url = kwargs['url']
    width = int(kwargs.get('width', 1024)) # screen width to capture
    height = int(kwargs.get('height', 768)) # screen height to capture
    filename = kwargs.get('filename', 'screen.png') # file name e.g. screen.png
    path = kwargs.get('path', ROOT) # directory path to store screen

    crop = kwargs.get('crop', False) # crop the captured screen
    crop_width = int(kwargs.get('crop_width', width)) # the width of crop screen
    crop_height = int(kwargs.get('crop_height', height)) # the height of crop screen
    crop_replace = kwargs.get('crop_replace', False) # does crop image replace original screen capture?

    thumbnail = kwargs.get('thumbnail', False) # generate thumbnail from screen, requires crop=True
    thumbnail_width = int(kwargs.get('thumbnail_width', width)) # the width of thumbnail
    thumbnail_height = int(kwargs.get('thumbnail_height', height)) # the height of thumbnail
    thumbnail_replace = kwargs.get('thumbnail_replace', False) # does thumbnail image replace crop image?

    screen_path = abspath(path, filename)
    crop_path = thumbnail_path = screen_path

    if thumbnail and not crop:
        raise Exception, 'Thumnail generation requires crop image, set crop=True'

    do_screen_capturing(url, screen_path, width, height)

    if crop:
        if not crop_replace:
            crop_path = abspath(path, 'crop_'+filename)
        params = {
            'width': crop_width, 'height': crop_height,
            'crop_path': crop_path, 'screen_path': screen_path}
        do_crop(params)

        if thumbnail:
            if not thumbnail_replace:
                thumbnail_path = abspath(path, 'thumbnail_'+filename)
            params = {
                'width': thumbnail_width, 'height': thumbnail_height,
                'thumbnail_path': thumbnail_path, 'crop_path': crop_path}
            do_thumbnail(params)
    return screen_path, crop_path, thumbnail_path



   
        #%%
dbname = 'frontpage'
username = 'dsaunder'

# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM srcs;"
srcs = pd.read_sql_query(sql_query,engine,index_col='index')

# Get the timestamp and setup directories
timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
frontpagedir = '../frontpages/%s/' % timestamp
if not os.path.exists(frontpagedir):
    os.makedirs(frontpagedir)

print "Downloading HTML web pages... "
# Download front page web pages HTML
for (i, src) in srcs.iterrows():
    response = requests.get(src['front_page'])
    if response.status_code == 200:    
        outfile = frontpagedir + src['prefix'] + timestamp + '.html'
        with open(outfile, 'w') as f:
            f.write(response.content)
    else:
        print "Failed to access URL %s" % src['front_page']
#%%
print "Downloading images of web pages... "
# Download front page web pages as images
for (i, src) in srcs.iterrows():
    url = src['front_page']
    screen_path, crop_path, thumbnail_path = get_screen_shot(
        url=url, filename=frontpagedir + src['prefix'] + timestamp + '.png',
        crop=True, crop_replace=True, crop_height=2304,
        thumbnail=True, thumbnail_replace=False, 
        thumbnail_width=200, thumbnail_height=150,
    )
#%%

