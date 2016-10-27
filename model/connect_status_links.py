#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Connect facebook statuses to scraped headline links.

Rough code that might be useful someday, e.g. to see which stories tend to get
cross-posted on facebook.

Created on Mon Sep 19 17:31:58 2016

@author: dsaunder
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import scipy
import time
import matplotlib.pyplot as plt


#%%
sql_query = "SELECT * FROM fb_matchstr_lookup;" # WHERE article_order <= 10;"
matchstr_lookup = pd.read_sql_query(sql_query,engine)
sql_query = "SELECT * FROM frontpage;" # WHERE article_order <= 10;"
frontpage_data = pd.read_sql_query(sql_query,engine)
#%%
# Find all the statuses that match frontpages in my database
# Note that there can sometimes be more than one facebook status per
# news story - in this case I will just pick the first one

fb_matches = []
nummatches = 0
for i,row in frontpage_data.iterrows():
     matches = [a in row.url for a in matchstr_lookup.loc[matchstr_lookup.src==row.src,'matchstr']]
     if (np.sum(matches) >= 1):
         for mappingindex in np.nonzero(matches)[0]:
             matching_status = matchstr_lookup.loc[matchstr_lookup.src==row.src].iloc[mappingindex]
#             print matching_status.status_id + "   " + matching_status.matchstr + "   " + matching_status.url
             fb_matches.append({'status_id':matching_status.status_id, 'fp_url':row.url, 'article_id':row.fp_timestamp+"-"+row.src+"-"+str(int(row.article_order))})
         nummatches = nummatches + 1

# fb_matches simply connects an article URL to a facebook status id
fb_matches = pd.DataFrame(fb_matches)

#%% Look up tweets for all the articles that I haven't already collected tweets
# for (necessary because in my normal script I only collect tweets for the 
# first 10 stories of the day)
urls_needed = np.unique(fb_matches.fp_url)
urls_needed_str = ["'%s'" % a for a in urls_needed]
urls_needed_str = ','.join(urls_needed_str)

sql_query = "SELECT url FROM tweet_download_log WHERE url IN (%s);" % urls_needed_str
urls_already_downloaded =  pd.read_sql_query(sql_query,engine)
urls_needed = [a for a in urls_needed if not a in urls_already_downloaded.url.values]
articles_needed = frontpage_data.merge(pd.DataFrame({'url':urls_needed}), on='url', how='right')

articles_needed.drop_duplicates(subset='url',inplace=True)
print len(articles_needed)


total_start_time = time.time()
new_tweet_list = []
try:
    for i,article in articles_needed.iterrows():
        new_tweet_list.extend(get_all_tweets_for_article(article, engine, ts))

except TwitterSearchException as e:
    print "Did not complete source %s" % src 
                
new_tweets = pd.DataFrame(new_tweet_list)
tweets_retrieved = len(new_tweets)


use_row = []
if len(new_tweets) > 0:
    
    sql_query = "SELECT id FROM tweets;" 
    
    existing_ids = pd.read_sql_query(sql_query,engine)
    existing_ids = set(existing_ids.values.flat)
    use_row = np.invert(new_tweets.id.isin(existing_ids))
    if np.sum(use_row) > 0:
        # Add the new tweets to the collection
        new_tweets.loc[use_row,:].to_sql('tweets', engine, if_exists='append')
        
print "Total time elapsed: %.1f minutes." % ((time.time() - total_start_time ) / 60.)
print "Total tweets retrieved: %d" % tweets_retrieved
print("Wrote %d new tweets to database" % np.sum(use_row))

#%%

# Before this next step, we've got to compute tweet negativities, 
# and article sis 

#%%
sql_query = 'SELECT * FROM fb_statuses;'
fb = pd.read_sql_query(sql_query,engine)

sql_query = "SELECT * FROM frontpage;"
frontpage_data =  pd.read_sql_query(sql_query,engine)
sql_query = "SELECT * FROM sis_for_articles;"
sis_for_articles =  pd.read_sql_query(sql_query,engine)

matching_urls = np.unique(fb_matches.fp_url)

# Prepare the list of articles that we have found to match
articles_to_compare = frontpage_data.merge(pd.DataFrame({'url':matching_urls}), on='url', how='right')
articles_to_compare = articles_to_compare.merge(sis_for_articles, on='url')

# Prepare the list of facebook stories we have found to match
fb_to_compare = fb_matches.merge(fb, on='status_id')
gb = fb_to_compare.groupby('fp_url')
fb_reactions_only = gb.sum()  # Deal with double postings on FB of the same article
fb_reactions_only.drop(['index','level_0'],1,inplace=True)
fb_reactions_only.loc[:,'fp_url'] = fb_reactions_only.index

# Now merge on the front page (newspaper website) URL
fb_tweet_comparison = articles_to_compare.merge(fb_to_compare.drop_duplicates(subset=['fp_url']), left_on='url',right_on='fp_url')
#fb_tweet_comparison = articles_to_compare.merge(fb_reactions_only, left_on='url',right_index=True, how='inner',suffixes=('l_','r_'))

# Get rid of duplicates (which can occur)
fb_tweet_comparison.drop_duplicates(subset=['url'],inplace=True)

# Compute some summaries
fb_tweet_comparison.loc[:,'prop_neg_fb'] = (fb_tweet_comparison.num_angries + fb_tweet_comparison.num_sads) / fb_tweet_comparison.num_reactions
fb_tweet_comparison.loc[:,'prop_angry_fb'] = fb_tweet_comparison.num_angries / fb_tweet_comparison.num_reactions
#sns.distplot(fb_tweet_comparison.loc[:,'prop_neg_fb'])

fb_tweet_comparison.loc[:,['headline','link_name','num_tweets','num_neg','prop_neg','sis','num_reactions','prop_angry_fb']].to_csv('fb_tweet_to_inspect.csv',encoding='utf-8')
#%%
import seaborn as sns
plt.figure()
sns.regplot(x='sis',y='prop_neg_fb', data=fb_tweet_comparison)
scipy.stats.pearsonr(fb_tweet_comparison.sis, fb_tweet_comparison.prop_neg_fb)
#%%
plt.figure()
sns.regplot(x='sis',y='prop_angry_fb', data=fb_tweet_comparison)
scipy.stats.pearsonr(fb_tweet_comparison.sis, fb_tweet_comparison.prop_angry_fb)
#%%
plt.figure()
sns.regplot(fb_tweet_comparison.sis, fb_tweet_comparison.prop_neg_fb)
sns.regplot(np.log(fb_tweet_comparison.sis), np.log(fb_tweet_comparison.prop_angry_fb))
sns.pairplot(fb_tweet_comparison)
sns.regplot()



fb_tweet_comparison.sort_values('zsis').loc[:,['headline','num_tweets','num_neg']]

#%%
g = sns.FacetGrid(fb_tweet_comparison, col="src")
g.map(sns.regplot, 'sis', 'prop_neg_fb')

#%%
plt.figure()
sns.regplot(x='sis',y='prop_neg_fb', data=fb_tweet_comparison)
#%%
plt.figure()
sns.regplot(x='sis',y='prop_angry_fb', data=fb_tweet_comparison)
#%%
plt.figure()
sns.regplot(x='sis',y='prop_angry_fb', data=fb_tweet_comparison.loc[fb_tweet_comparison.src =='bbc',:])

fb_tweet_comparison.loc[fb_tweet_comparison.src =='bbc',['headline','sis','prop_neg_fb']].sort_values('prop_neg_fb')

#%%
# What's going on with multiple stories?
fb_story_count = gb.count().src
fb_story_multiple = fb_story_count.loc[fb_story_count>1]
inspect_multiples = fb_to_compare.join(pd.DataFrame(fb_story_multiple),how='right',on='fp_url',lsuffix='',rsuffix='r')
inspect_multiples.sort_values('fp_url').loc[:,['fp_url','status_message','num_likes']]
#%%
g = sns.FacetGrid(fb_tweet_comparison, col="src")
g.map(sns.regplot, 'num_tweets', 'num_reactions')


fb_tweet_comparison.to_csv('fb_tweet_comparison.csv',encoding='utf-8')

