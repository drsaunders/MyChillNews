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
import extract_headlines  # LOCAL MODULE
import compute_sis_with_model # LOCAL MODULE
import numpy as np
import getpass


#%%
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
            'front_page':'http://www.nytimes.com/'},
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
            'front_page':'https://www.theguardian.com/uk?INTCMP=CE_UK'},
             {'name':'The Boston Globe', 
             'prefix':'bos',
            'fb_page':'globe',
            'front_page':'http://www.bostonglobe.com'}
           ]
    srcs = pd.DataFrame(srcs)
    srcs.to_sql('srcs', engine, if_exists='replace')

#%%




def extract_headlines_to_db(fp_timestamp, engine):
    new_headlines = extract_headlines.extract_all_headlines(fp_timestamp)
    sql_query = "SELECT article_id FROM frontpage;"
    existing_ids = pd.read_sql_query(sql_query,engine)
    existing_ids = set(existing_ids.values.flat)
    use_row = np.invert(new_headlines.article_id.isin(existing_ids))
    new_headlines.loc[use_row,:].to_sql('frontpage', engine, if_exists='append')
    print("Wrote %d new headlines to database" % np.sum(use_row))
    

def add_article_id_to_db(engine):
    sql_query = "SELECT * FROM frontpage;" # WHERE article_order <= 10;"
    frontpage_data = pd.read_sql_query(sql_query,engine)
    frontpage_data.loc[:,'article_id'] = [a.fp_timestamp+"-"+a.src+"-"+str(int(a.article_order)) for i,a in frontpage_data.iterrows()]
    frontpage_data.to_sql('frontpage', engine, if_exists='replace')
    frontpage_data.to_sql('frontpage', engine, if_exists='replace', chunksize=100)

#%%
if __name__ == '__main__':
    
    abspath = lambda *p: os.path.abspath(os.path.join(*p))
    ROOT = abspath(os.path.dirname(__file__))
    
    dbname = 'frontpage'
    username = getpass.getuser()
    if username == 'root':  # Hack just for my web server
        username = 'ubuntu'
    
       
    # prepare for database
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    
    sql_query = "SELECT * FROM srcs;"
    srcs = pd.read_sql_query(sql_query,engine,index_col='index')
    
    # Get the timestamp and setup directories
    fp_timestamp =  datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    #fp_timestamp = '2016-09-22-0724'
    frontpagedir = '../current_frontpage/'
    if not os.path.exists(frontpagedir):
        os.makedirs(frontpagedir)
    
    # Download front page web pages HTML
    print "Downloading HTML web pages... "
    for (i, src) in srcs.iterrows():
        response = requests.get(src['front_page'])
        if response.status_code == 200:    
            outfile = frontpagedir + src['prefix'] + '.html'
            with open(outfile, 'w') as f:
                f.write(response.content)
        else:
            print "Failed to access URL %s" % src['front_page']
    #%%
    print "Extracting headlines... "
    extract_headlines_to_db(fp_timestamp, engine)
    
    #%%
    print "Computing stress impact scores for articles..."
    compute_sis_with_model.compute_sis_for_all(engine)
