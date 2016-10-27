#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:14:32 2016

@author: dsaunder
"""

import requests
import os
import datetime
import pandas as pd
from sqlalchemy import create_engine
from subprocess import Popen, PIPE
from selenium import webdriver
import extract_headlines  # LOCAL MODULE
import numpy as np
import getpass

from subprocess import Popen, PIPE
from selenium import webdriver
from sqlalchemy import create_engine

def do_screen_capturing(url, screen_path, width, height):
    print "Capturing screen.."
    driver = webdriver.PhantomJS(service_args=['--ignore-ssl-errors=true', '--ssl-protocol=ANY'])
    print "Initialized driver"
    # it save service log file in same directory
    # if you want to have log file stored else where
    # initialize the webdriver.PhantomJS() as
    # driver = webdriver.PhantomJS(service_log_path='/var/log/phantomjs/ghostdriver.log')
    driver.set_script_timeout(220)
    if width and height:
        driver.set_window_size(width, height)
    print url
    driver.get(url)
    print "Got URL"
    print screen_path
    driver.save_screenshot(screen_path)
    print "Saved screenshot"

def execute_command(command):
    result = Popen(command, shell=True, stdout=PIPE).stdout.read()
    if len(result) > 0 and not result.isspace():
        raise Exception(result)

def do_crop(params):
    command = [
        'convert',
        params['screen_path'],
        '-crop', '%sx%s+0+0' % (params['width'], params['height']),
        params['crop_path']
    ]
#    print command
    execute_command(' '.join(command))
#    print ' '.join(command)


def do_thumbnail(params):
    print "Generating thumbnail from cropped captured image.."
    params2 = {'screen_path':params['crop_path'], 
    'crop_path':params['thumbnail_path'], 
        'width': params['crop_width'], 'height': params['crop_height']/2}
    do_crop(params2)
    command = [
        'convert',
        params['thumbnail_path'],
        '-filter', 'Lanczos',
        '-thumbnail', '%sx%s' % (params['width'], params['height']),
        params['thumbnail_path']
    ]
    execute_command(' '.join(command))
#    print ' '.join(command)

def get_screen_shot(**kwargs):
    url = kwargs['url']
    width = int(kwargs.get('width', 1024)) # screen width to capture
    height = int(kwargs.get('height', 768)) # screen height to capture
    filename = kwargs.get('filename', 'screen.png') # file name e.g. screen.png
    path = kwargs.get('path', ROOT) # directory path to store screen

    crop = kwargs.get('crop', False) # crop the captured screen
    crop_width = int(kwargs.get('crop_width', width)) # the width of crop screen
    crop_height = int(kwargs.get('crop_height', height)) # the height of crop screen
    crop_replace = kwargs.get('crop_replace', False) # does crop image replace original screen capture?

    thumbnail = kwargs.get('thumbnail', False) # generate thumbnail from screen, requires crop=True
    thumbnail_width = int(kwargs.get('thumbnail_width', width)) # the width of thumbnail
    thumbnail_height = int(kwargs.get('thumbnail_height', height)) # the height of thumbnail
    thumbnail_replace = kwargs.get('thumbnail_replace', False) # does thumbnail image replace crop image?
    thumbnail_filename = kwargs.get('thumbnail_filename', ROOT) # does thumbnail image replace crop image?

    screen_path = abspath(path, filename)
    crop_path = thumbnail_path = screen_path
    
    if thumbnail and not crop:
        raise Exception, 'Thumnail generation requires crop image, set crop=True'

    do_screen_capturing(url, screen_path, width, height)

    if crop:
        if not crop_replace:
            crop_path = abspath(path, 'crop_'+filename)
        params = {
            'width': crop_width, 'height': crop_height,
            'crop_path': crop_path, 'screen_path': screen_path}
        do_crop(params)

        if thumbnail:
            if not thumbnail_replace:
                thumbnail_path = abspath(path, "thumbnail_" + filename)
            params = {
                'width': thumbnail_width, 'height': thumbnail_height,
                'thumbnail_path': thumbnail_path, 'crop_path': crop_path, 'crop_width':crop_width, 'crop_height':crop_height}
            do_thumbnail(params)
    return screen_path, crop_path, thumbnail_path

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
    
    frontpagedir = 'app/frontpage/static/current_frontpage_thumbnails/'
    print "Downloading images of web pages... "
    for (i, src) in srcs.iterrows():
        url = src['front_page']
        screen_path, crop_path, thumbnail_path = get_screen_shot(
            path=frontpagedir,      
            url=url, filename=src['prefix'] + '.png',
            crop=True, crop_replace=True, crop_height=2304,
            thumbnail=True, thumbnail_replace=False, 
            thumbnail_width=400, thumbnail_height=350,
        )
        
        os.system('killall phantomjs')

    # Copy the new thumbnails to my web server, overwriting the old ones    
    execstr = 'scp -i ../insight2016.pem %sthumbnail*.png ubuntu@52.43.167.177:/home/ubuntu/FrontPage/app/frontpage/static/current_frontpage_thumbnails/' % frontpagedir
    print os.system(execstr)
    
