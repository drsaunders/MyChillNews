#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Makes wordclouds of top 1000 and bottom 1000 most angry headlines.
Save them as two pngs.

Created on Wed Sep 28 15:05:30 2016

@author: dsaunder
"""

import wordcloud
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine
import getpass

from wordcloud import WordCloud
def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    the_color = sns.color_palette("coolwarm",n_colors=2)[0]
    return (255,0,0)

def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    the_color = sns.color_palette("coolwarm",n_colors=2)[1]
    return (0,0,255)
    
# Make a fancy ellipse mask for the cloud
width = 2000
height = 1600
x,y = np.meshgrid(range(width),range(height))
ellipse_mask = ((x-1000)/1000.)**2 + ((y-800)/800.)**2 > 1
ellipse_mask = ellipse_mask.astype(int)*255
 #%%
    
dbname = 'frontpage'
username = getpass.getuser()
if username == 'root':  # Hack just for my web server
    username = 'ubuntu'

   
# prepare for database
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

# Use SIS for all headlines in the database
sql_query = """                                                             
            SELECT headline, src, sis, frontpage.url as url, frontpage.article_id as article_id FROM frontpage 
            JOIN srcs ON frontpage.src=srcs.prefix
            JOIN sis_for_articles_model ON frontpage.article_id = sis_for_articles_model.article_id
            """
frontpages = pd.read_sql_query(sql_query,engine)
frontpages.drop_duplicates(subset=['url'],inplace=True)
frontpages.drop_duplicates(subset=['headline'],inplace=True)
#%%
# Create and export word cloud for the least stressful 1000 headlines
headlines = frontpages.headline
sis = frontpages.sis
lower_text = ' '.join(headlines.iloc[np.argsort(sis)[:1000]])
wordcloud = WordCloud(mask=ellipse_mask, background_color='white', relative_scaling=1,width=2000,height=1600, min_font_size=30).generate(lower_text)
wordcloud.recolor(0,blue_color_func)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('lower.png')

#%%
# What's up with "new" appearing in all these low SIS headlines?
for h in headlines.iloc[np.argsort(sis)[:1000]]:
    if 'new' in h.lower():
        print h
        
#%%
# Create and export word cloud for the most stressful 1000 headlines

upper_text = ' '.join(headlines.iloc[np.argsort(sis)[-1000:]])

wordcloud = WordCloud(mask=ellipse_mask, background_color='white', relative_scaling=1,width=2000,height=1600, min_font_size=30).generate(upper_text)
#wordcloud = WordCloud(background_color='white', relative_scaling=0.1, max_words=100).generate(upper_text)
wordcloud.recolor(0,red_color_func)
plt.figure(figsize=(10,10))
#wordcloud = WordCloud(background_color='white').generate(lower_text)
#plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('upper.png')
plt.figure()
plt.imshow(ellipse_mask)
