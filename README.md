# MyChillNews

This is a data-driven app that I prototyped in 3 weeks as part of my fellowship with [Insight Health Data Science](http://insighthealthdata.com/). The live app should be accessible at [MyChillNews.co](http://mychillnews.co)

![MyChillNews app screenshot](app_screenshot.png)

I trained a model on headlines and Facebook API reaction data, to be able to predict the stressfulness of news stories, using scikit-learn and NLTK. 

Every day, morning and noon, my script scrapes the top headlines from 10 top news stories, and stores both them and their score according to the model in a PostgreSQL database. These Stress Impact Scores are averaged for different news sources, and then when users access the app they are shown the names of the news sources coloured by their predicted stress impact.

The purpose of the app is to help people who like to read daily news become conscious consumers with regard to the stressful impact of the stories they read. I think that starting your day with MyChillNews is one of those things that feels good, and is also good for you.
