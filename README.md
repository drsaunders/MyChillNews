# MyChillNews

This is a data-driven app that I prototyped in 3 weeks as part of my fellowship with [Insight Health Data Science](http://insighthealthdata.com/). The live app should be accessible at [MyChillNews.co](http://mychillnews.co)

![MyChillNews app screenshot](app_screenshot.png)

I trained a model on headlines and Facebook API reaction data, to be able to predict the stressfulness of news stories, using scikit-learn and NLTK. 

Every day, morning and noon, my script scrapes the top headlines from 10 top news stories, and stores both them and their score according to the model in a PostgreSQL database running on an AWS EC2 instance. These Stress Impact Scores are averaged for different news sources, and then when users access the app they are shown the names of the news sources coloured by their predicted stress impact. They can then adjust both the preference order of the news sources (with a reorderable JavaScript menu provided by the Sortable library) and set their maximum Stress Impact Score for the day, resulting in a dynamically updated recommendation of what they should read today. 

The purpose of the app is to assist people who like to read daily news with conscious consumption of stressful news stories. I think that starting your day with MyChillNews is one of those things that feels good, and is also good for you.

The codebase consists of two parts:

* `model` This code relates to "behind the scenes" operations: downloading Facebook data, training the model, and updating the database with today's headlines and their predicted stress impacts.
* `app` This is a Flask application, using Bootstrap, which accesses the database to render the MyChillNews app. It is normally running on an EC2 instance, the server being gunicorn managed by supervisord. Most of the logic is contained in `app/frontpage/views.py`.
