#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:17:30 2016

@author: dsaunder
"""

client_id = '227YZJ'
consumer_secret = '441fffda0c697b29077376af97d2d3ad'


access_token = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIzWlM5UEciLCJhdWQiOiIyMjdZWkoiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNDczOTEyNDE2LCJpYXQiOjE0NzM4ODM2MTZ9.77vUL9ZL9-jOa9UYRYuRTRsPfxgq24m9Mn4BNvWCD5A'
refresh_token = 'eba241fada1dde5cf18dee6cf2c345e9601458c831d35576483baea410149a9b'

authd_client = fitbit.Fitbit(client_id, consumer_secret,
                             access_token=access_token, refresh_token=refresh_token)
authd_client.heart()

client = fitbit.FitbitOauth2Client(client_id, consumer_secret)
token = client.fetch_request_token()
print 'FROM RESPONSE'
print 'key: %s' % str(token.key)
print 'secret: %s' % str(token.secret)
print 'callback confirmed? %s' % str(token.callback_confirmed)
print ''