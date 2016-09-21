from flask import render_template
from frontpage import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
import datetime
import seaborn as sns
import numpy as np
import re
def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor

max_sis = 0.16
color_range = sns.color_palette("coolwarm",n_colors=int(max_sis*1000)) #sns.color_palette("RdBu_r",n_colors=151)
color_range = [(a[0]*255, a[1]*255, a[2]*255) for a in color_range]
hex_colors = np.array([RGBToHTMLColor(rgb_tuple) for rgb_tuple in color_range])
user = 'dsaunder' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'frontpage'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
#todays_date = '2016-09-15'
#%%

@app.route('/')
@app.route('/index')
def index():
    sql_query = """                                                             
                SELECT * FROM frontpage 
                JOIN srcs ON frontpage.src=srcs.prefix
                JOIN sis_for_articles ON frontpage.url = sis_for_articles.url
                WHERE fp_timestamp LIKE '%s%%' AND article_order <= 10;                                                                               
                """  % todays_date
    frontpage_for_render = pd.read_sql_query(sql_query,con)
    
    by_name = frontpage_for_render.loc[frontpage_for_render.num_tweets > 5,:].groupby('name')
    mean_by_name = by_name.mean()
    total_by_name = by_name.sum()
    row_colors = hex_colors[np.round(mean_by_name.sis.values*1000).astype(int)]

    src_names_string = ','.join(['"%s"' % a for a in mean_by_name.index.values])
    sis_values_string = ','.join(['%.1f' % (a*1000) for a in mean_by_name.sis.values])
    url_list = [frontpage_for_render.loc[frontpage_for_render.name ==a,'front_page'].iloc[0] for a in mean_by_name.index.values]
    url_string = ','.join('"%s"' % a for a in url_list)
    return render_template("index.html",
       todays_date = todays_date,
       total_num_tweets = np.sum(frontpage_for_render.num_tweets),
       mean_by_name = mean_by_name,
       total_by_name = total_by_name,
       src_names_string=src_names_string,
       sis_values_string=sis_values_string,
       url_string = url_string,
       row_colors=row_colors,
       )
