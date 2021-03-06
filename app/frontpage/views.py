"""
Flask module to render the main page of the app, using stored database info.

Uses Stress Impact Scores (and percentiles) for today's new source front pages
to set up the colours of the list of news sources, as well as the 
"""

from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import seaborn as sns
import numpy as np
import getpass

from flask import render_template
from flask import render_template_string
from flask import request
#%%
from frontpage import app
#%%

def RGBToHTMLColor(rgb_tuple):
    """ Convert an (R, G, B) tuple to #RRGGBB """
    hexcolor = '#%02x%02x%02x' % rgb_tuple

    return hexcolor

# Set up the range of colours to use for the source list, and convert to HTML hex colours
color_range = sns.color_palette("coolwarm",n_colors=100) #sns.color_palette("RdBu_r",n_colors=151)
color_range = [(a[0]*255, a[1]*255, a[2]*255) for a in color_range]
hex_colors = np.array([RGBToHTMLColor(rgb_tuple) for rgb_tuple in color_range])

# Prepare for the database
user = getpass.getuser()
if user == 'root':  # Hack just for my web server
    user = 'ubuntu'
host = 'localhost'
dbname = 'frontpage'

date_to_use = None
#%%

@app.route('/')
@app.route('/index')
def index():
    # Can request to see a particular date by passing in the date in the url,
    # e.g. http://www.mychillnews.co/index?date=2016-10-26'
    date_to_use = request.args.get('date')
#%%
    db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
    con = None
    con = psycopg2.connect(database = dbname, user = user)

    # If no date specified, use the most recent day that has SIS data
    if date_to_use is None:
        date_query = """
            SELECT fp_timestamp
            FROM frontpage JOIN sis_for_articles_model
                ON frontpage.article_id = sis_for_articles_model.article_id
            ORDER BY frontpage.article_id DESC LIMIT 1;                """
        date_to_use = pd.read_sql_query(date_query,con).values[0][0]

    # Get information about the articles and their SISs
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
    #%%
    # If nothing found, display error message
    if len(frontpage_for_render) == 0:
        return render_template_string('No data for date %s' % date_to_use)
        #%%
    frontpage_for_render.drop_duplicates(subset=['url'],inplace=True)

    # For each news source, take the mean of the SIS percentiles to be the aggregate
    # SIS.
    by_name = frontpage_for_render.groupby(['name','src'])
    mean_by_name = by_name.mean().reset_index()
    total_by_name = by_name.sum().reset_index()
    sis_for_frontpages = mean_by_name.sis_pct.values

    # Rescale to try to make good use of the range, otherwise the means are too close to 0.5 
    # and the colours are tepid.
    # Important that this is a constant however, so days can be compared
    sis_for_frontpages = (sis_for_frontpages - 0.5)*2+0.5
    sis_for_frontpages[sis_for_frontpages>0.98] = 0.98
    sis_for_frontpages[sis_for_frontpages<0.02] = 0.02

    # Build the list of news sources, along with the colours they should be rendered in
    src_names_string = ','.join(['"%s"' % a for a in mean_by_name.name.values])
    sis_values_string = ','.join(['%.1f' % (a*1000) for a in sis_for_frontpages])
    row_colors = hex_colors[np.floor(sis_for_frontpages*100).astype(int)]

    # Build the list of news source URLS
    url_list = [frontpage_for_render.loc[frontpage_for_render.name ==a,'front_page'].iloc[0] for a in mean_by_name.name.values]
    url_string = ','.join('"%s"' % a for a in url_list)

    # Build the list of thumbnail files
    thumbnail_paths = ['/static/current_frontpage_thumbnails/thumbnail_%s.png' % frontpage_for_render.loc[frontpage_for_render.name ==a,'src'].iloc[0] for a in mean_by_name.name.values]
    thumbnail_string = ','.join('"%s"' % a for a in thumbnail_paths)

    # Write out info about individual article scores to help with debugging the scoring decisions
    with open("frontpage_scoring.txt",'w') as fid:
        fid.write('Computation of Stress Impact Scores for today''s news sources')
        
        for i in range(len(sis_for_frontpages)):
            fid.write("\n\n%s  frontpage SIS:  %.1f (rescaled for color, 0-1: %.2f)" % (mean_by_name.name.values[i], mean_by_name.sis_pct.values[i]*100, sis_for_frontpages[i]))
            for_src = frontpage_for_render.loc[frontpage_for_render.src == mean_by_name.src.values[i],:].copy()
            for_src.sort_values(['src','sis'],inplace=True)
            for j,h in for_src.iterrows():
                fid.write("\n\t%.1f %s" % (h.sis_pct*100, h.headline))

#    frontpage_for_render.loc[:,['src','headline','sis','sis_pct']].sort_values(['src','sis']).to_csv('frontpage_scoring.csv',encoding='utf-8')

#%%
    # Build the web page from the index.html template
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

@app.route('/contact')
def contact():
    return render_template("contact.html")

# Feature to see where the day's score comes from
@app.route('/why')
def why():
    with open("frontpage_scoring.txt",'r') as fid:
        why_text = fid.read().decode("utf8")
        
    return render_template("why.html", why_text=why_text)
    
# Fun feature to query the most horrible headline of each day from the daily 
# mail, mychillnews.co/dm
@app.route('/dm')
def dm():
    #%%
    sql_query = """
    SELECT min(frontpage.fp_timestamp) as timestamp, to_char(AVG(sis_pct),'0.9999') as avg_sis, headline FROM frontpage 
        JOIN sis_for_articles_model 
        ON frontpage.article_id = sis_for_articles_model.article_id
        JOIN 
            (SELECT fp_timestamp, src, max(sis_pct) as max_sis FROM frontpage JOIN 		
             sis_for_articles_model ON frontpage.article_id = sis_for_articles_model.article_id 	
             WHERE frontpage.src='dm' GROUP BY fp_timestamp, src) as maxes
        ON frontpage.src = maxes.SRC AND sis_for_articles_model.sis_pct = max_sis AND frontpage.fp_timestamp = maxes.fp_timestamp
    GROUP BY headline
    ORDER BY timestamp
    """
    con = psycopg2.connect(database = dbname, user = user)
    dailymail_headlines = pd.read_sql_query(sql_query,con)
    old_colwidth = pd.options.display.max_colwidth
    pd.options.display.max_colwidth = 1000
    outstr = "Date" + " "*13 + "SIS" + " "*4 + "Headline\n"
    for i, row in dailymail_headlines.iterrows():
        outstr = outstr + "%s\t%s\t%s\n" % (row['timestamp'], row['avg_sis'], row['headline'].decode("utf8"))
    return render_template("dm.html", dm_text=outstr)
    pd.options.display.max_colwidth = old_colwidth
    con.close()
    