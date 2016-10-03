from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import datetime
import seaborn as sns
import numpy as np
import re
import os
import getpass

#%%
from flask import render_template
from flask import render_template_string
from frontpage import app
from flask import request
#%%

def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor

color_range = sns.color_palette("coolwarm",n_colors=100) #sns.color_palette("RdBu_r",n_colors=151)
color_range = [(a[0]*255, a[1]*255, a[2]*255) for a in color_range]
hex_colors = np.array([RGBToHTMLColor(rgb_tuple) for rgb_tuple in color_range])
user = getpass.getuser()
if user == 'root':  # Hack just for my web server
    user = 'ubuntu'
 #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'frontpage'

todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
#todays_date = '2016-09-28'
#%%


@app.route('/')
@app.route('/index')
def index():
#%%
    db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
    con = None
    con = psycopg2.connect(database = dbname, user = user)
#     Use the most recent day that has sis data 
    date_to_use = request.args.get('date')
    if date_to_use is None:
        date_query = """                                                             
            SELECT fp_timestamp 
            FROM frontpage JOIN sis_for_articles_model 
                ON frontpage.article_id = sis_for_articles_model.article_id 
            ORDER BY frontpage.article_id DESC LIMIT 1;                """  
        date_to_use = pd.read_sql_query(date_query,con).values[0][0]

    sql_query = """                                                             
                SELECT article_order
                    , fp_timestamp
                    , headline
                    , src
                    , frontpage.url as url
                    , frontpage.article_id as article_id
                    , name
                    , sis
                    , sis_pct
                    , front_page
                    , zsis
                    as article_id FROM frontpage 
                JOIN srcs ON frontpage.src=srcs.prefix
                JOIN sis_for_articles_model ON frontpage.url = sis_for_articles_model.url
                WHERE fp_timestamp LIKE '%s%%' AND article_order <= 10;                                                                               
                """  % date_to_use
    frontpage_for_render = pd.read_sql_query(sql_query,con)
    con.close()
    
    if len(frontpage_for_render) == 0:
        return render_template_string('No data for date %s' % date_to_use)
#%%
    frontpage_for_render.drop_duplicates(subset=['url'],inplace=True)

    by_name = frontpage_for_render.groupby('name')
    mean_by_name = by_name.mean()
    total_by_name = by_name.sum()
    sis_for_frontpages = mean_by_name.sis_pct.values

    # Adjust 
    sis_for_frontpages = (sis_for_frontpages - 0.5)*np.sqrt(10)+0.5
    sis_for_frontpages[sis_for_frontpages>0.98] = 0.98
    sis_for_frontpages[sis_for_frontpages<0.02] = 0.02


    row_colors = hex_colors[np.floor(sis_for_frontpages*100).astype(int)]

    src_names_string = ','.join(['"%s"' % a for a in mean_by_name.index.values])
    sis_values_string = ','.join(['%.1f' % (a*1000) for a in sis_for_frontpages])

    url_list = [frontpage_for_render.loc[frontpage_for_render.name ==a,'front_page'].iloc[0] for a in mean_by_name.index.values]
    url_string = ','.join('"%s"' % a for a in url_list)
    print os.getcwd()
    thumbnail_paths = ['/static/current_frontpage_thumbnails/thumbnail_%s.png' % frontpage_for_render.loc[frontpage_for_render.name ==a,'src'].iloc[0] for a in mean_by_name.index.values]
    thumbnail_string = ','.join('"%s"' % a for a in thumbnail_paths)
    
    
    frontpage_for_render.loc[:,['src','headline','sis','sis_pct']].sort_values(['src','sis']).to_csv('frontpage_scoring.csv',encoding='utf-8')
#%%
    return render_template("index.html"
       ,date_to_use = '%s %s:%s' % (date_to_use[:-5], date_to_use[-4:-2], date_to_use[-2:])
       ,total_num_tweets = 0
       ,mean_by_name = mean_by_name
       ,total_by_name = total_by_name
       ,src_names_string=src_names_string
       ,sis_values_string=sis_values_string
       ,url_string = url_string
       ,row_colors=row_colors
       ,thumbnail_string=thumbnail_string
       )

@app.route('/about')
def about():
    return render_template("about.html")