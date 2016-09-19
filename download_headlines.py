#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:36:36 2016

@author: dsaunder
"""

# Download the front pages and extract the titles and URLs of all their top headlines

import requests
import os
import datetime
import pandas as pd
from sqlalchemy import create_engine
from subprocess import Popen, PIPE
from selenium import webdriver
import extract_headlines  # LOCAL MODULE

from subprocess import Popen, PIPE
from selenium import webdriver
from sqlalchemy import create_engine

abspath = lambda *p: os.path.abspath(os.path.join(*p))
ROOT = abspath(os.path.dirname(__file__))
#%%



frontpagedir = '../frontpages/%s/' % timestamp
dbname = 'frontpage'
username = 'dsaunder'
   
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))


def clean_df_db_dups(df, tablename, engine, dup_cols=[],
                         filter_continuous_col=None, filter_categorical_col=None):
    """
    Remove rows from a dataframe that already exist in a database
    Required:
        df : dataframe to remove duplicate rows from
        engine: SQLAlchemy engine object
        tablename: tablename to check duplicates in
        dup_cols: list or tuple of column names to check for duplicate row values
    Optional:
        filter_continuous_col: the name of the continuous data column for BETWEEEN min/max filter
                               can be either a datetime, int, or float data type
                               useful for restricting the database table size to check
        filter_categorical_col : the name of the categorical data column for Where = value check
                                 Creates an "IN ()" check on the unique values in this column
    Returns
        Unique list of values from dataframe compared to database table
    """
    args = 'SELECT %s FROM %s' %(', '.join(['"{0}"'.format(col) for col in dup_cols]), tablename)
    args_contin_filter, args_cat_filter = None, None
    if filter_continuous_col is not None:
        if df[filter_continuous_col].dtype == 'datetime64[ns]':
            args_contin_filter = """ "%s" BETWEEN Convert(datetime, '%s')
                                          AND Convert(datetime, '%s')""" %(filter_continuous_col,
                              df[filter_continuous_col].min(), df[filter_continuous_col].max())


    if filter_categorical_col is not None:
        args_cat_filter = ' "%s" in(%s)' %(filter_categorical_col,
                          ', '.join(["'{0}'".format(value) for value in df[filter_categorical_col].unique()]))

    if args_contin_filter and args_cat_filter:
        args += ' Where ' + args_contin_filter + ' AND' + args_cat_filter
    elif args_contin_filter:
        args += ' Where ' + args_contin_filter
    elif args_cat_filter:
        args += ' Where ' + args_cat_filter

    df.drop_duplicates(dup_cols, keep='last', inplace=True)
    df = pd.merge(df, pd.read_sql(args, engine), how='left', on=dup_cols, indicator=True)
    df = df[df['_merge'] == 'left_only']
    df.drop(['_merge'], axis=1, inplace=True)
    return df

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
    print "Generating thumbnail from cropped captured image.."
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
                thumbnail_path = abspath(path, filename + "_thumbnail")
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
fp_timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
frontpagedir = '../frontpages/%s/' % fp_timestamp
if not os.path.exists(frontpagedir):
    os.makedirs(frontpagedir)

# Download front page web pages HTML
print "Downloading HTML web pages... "
for (i, src) in srcs.iterrows():
    response = requests.get(src['front_page'])
    if response.status_code == 200:    
        outfile = frontpagedir + src['prefix'] + fp_timestamp + '.html'
        with open(outfile, 'w') as f:
            f.write(response.content)
    else:
        print "Failed to access URL %s" % src['front_page']
#%%
# Download front page web pages as images

print "Downloading images of web pages... "
for (i, src) in srcs.iterrows():
    url = src['front_page']
    screen_path, crop_path, thumbnail_path = get_screen_shot(
        url=url, filename=frontpagedir + src['prefix'] + fp_timestamp + '.png',
        crop=True, crop_replace=True, crop_height=2304,
        thumbnail=True, thumbnail_replace=False, 
        thumbnail_width=200, thumbnail_height=150,
    )

#%%
new_headlines = extract_headlines.extract_all_headlines(fp_timestamp)
sql_query = "SELECT url FROM frontpage;"
existing_ids = pd.read_sql_query(sql_query,engine)
existing_ids = set(existing_ids.values.flat)
use_row = np.invert(new_headlines.url.isin(existing_ids))
new_headlines.loc[use_row,:].to_sql('frontpage', engine, if_exists='append')
print("Wrote %d new headlines to database" % np.sum(use_row))