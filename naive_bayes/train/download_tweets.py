#!/usr/bin/python

import sys
import urllib
import re
import json
import urllib.request

from bs4 import BeautifulSoup

import socket
socket.setdefaulttimeout(10)

cache = {}

for line in open(sys.argv[1]):
    fields = line.rstrip('\n').split('\t')
    sid = fields[0]
    sen = fields[1]

    #url = 'http://twitter.com/intent/retweet?tweet_id=%s' % (sid)
    #print url
    tweet = None
    text = "Not Available"
    if sid in cache:
        text = cache[sid]
    else:
        try:
            url = "http://twitter.com/intent/retweet?tweet_id=" + sid
            f = urllib.request.urlopen(url)
            #Thanks to Aniruddha
            html = f.read()
            #html = html.replace("</html>", "") + "</html>"
            #print (html)
            BeautifulSoup(html, "lxml")
            jstt = soup.find_all("div","tweet-text")
            tweets = list(set([x.get_text() for x in jstt]))
            #print len(tweets)
            #print (tweets)
            #print ("\t")(tweets)
            print (sid + "\t" + sen + "\t" + tweets[0]).encode('utf-8')
        except Exception:
            continue
