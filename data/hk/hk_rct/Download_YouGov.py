## Note: This code is not compatible with Python 3.0.  Please use Python 2.7
## to run this code.
import requests
import json
import numpy as np
import pandas as pd


## This code gets the data from YouGov as a JSON.  When the authors ran the code,
## they used all available Tweets.  To replicate the analysis, it would be
## necessary to prune all Tweets from after 10/31/2017.
url = 'https://9p8f5cw3u7-dsn.algolia.net/1/indexes/tweets/query?x-algolia-agent=Algolia%20for%20vanilla%20JavaScript%203.21.1&x-algolia-application-id=9P8F5CW3U7&x-algolia-api-key=3ecdf0a9123006c6ccae27125cd090aa'
resp = requests.post(url, data=json.dumps({
'params': 'query=&hitsPerPage=500&page=0'
}))


doc = resp.json()

## This code merged the data with the text of Trump's tweets, but it no longer
## workds due to Twitter's deletion of Trump's account.
out = open('/Users/justingrimmer/Dropbox/SuperExp/LongPaper/TrumpTweet/Date.csv', 'w')
out.write('year,month,day')
out.write('\n')

for z in doc['hits']:
	temp = z['survey_date']
	year = str(temp)[:4]
	month = str(temp)[4:6]
	day = str(temp)[6:]
	out.write('%s,%s,%s' %(year,month,day))
	out.write('\n')
