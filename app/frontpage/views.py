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
def RGBToHTMLColor(rgb_tuple):
    """ convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple
    # that's it! '%02x' means zero-padded, 2-digit hex values
    return hexcolor

color_range = sns.color_palette("coolwarm",n_colors=151) #sns.color_palette("RdBu_r",n_colors=151)
color_range = [(a[0]*255, a[1]*255, a[2]*255) for a in color_range]
hex_colors = np.array([RGBToHTMLColor(rgb_tuple) for rgb_tuple in color_range])
user = 'dsaunder' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'frontpage'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
#%%

@app.route('/')
@app.route('/index')
def index():
    sql_query = """                                                             
                SELECT * FROM frontpage JOIN srcs ON frontpage.src=srcs.prefix WHERE fp_timestamp LIKE '%s%%' AND all_tweets_collected=TRUE;                                                                               
                """  % todays_date
    frontpage_for_render = pd.read_sql_query(sql_query,con)
    
    
    by_name = frontpage_for_render.loc[frontpage_for_render.num_tweets > 5,:].groupby('name')
    mean_by_zsis = by_name.mean() #.sort_values('zsis', ascending=False)
    total_by_zsis = by_name.sum()#.sort_values('zsis', ascending=False)
    row_colors = hex_colors[np.round(mean_by_zsis.sis.values*1000).astype(int)]

    src_names_string = ','.join(['"%s"' % a for a in mean_by_zsis.index.values])
    sis_values_string = ','.join(['%.1f' % (a*1000) for a in mean_by_zsis.sis.values])
    return render_template("index.html",
       mean_by_zsis = mean_by_zsis,
       total_by_zsis = total_by_zsis,
       src_names_string=src_names_string,
       sis_values_string=sis_values_string,
       row_colors=row_colors,
       )
