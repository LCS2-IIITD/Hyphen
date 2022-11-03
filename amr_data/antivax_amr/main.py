import pandas as pd
df = pd.read_csv("Labeled/VaxMisinfoData.csv")
from datetime import timedelta
from ratelimit import limits
import requests

import requests
import json
import time
# importing module
import logging
 
# Create and configure logger
logging.basicConfig(filename="antivax.log", format='%(asctime)s %(message)s',filemode='w')
 
# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
 
# Replace your own bearer token below from the academic research project in the Twitter developer portal
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAHrjXQEAAAAAzi7xGphFmwHPwMZTyWzCLH2xW9I%3DVr875clIIOT0fBXeXofXN6yvWb9aXG8HnJvLTKA07ho3NVOVDo'

search_url = "https://api.twitter.com/2/tweets/search/all"

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers, params):
    response = requests.request("GET",
                                search_url,
                                headers=headers,
                                params=params)
    
    if response.status_code == 429:
        a = int(response.headers["Retry-After"])
        print("Waiting for {} seconds".format(a))
        time.sleep(a)
        response = requests.request("GET", search_url, headers=headers, params=params)
        
    if response.status_code != 200:
        print(response.status_code)
        raise Exception(response.status_code, response.text)
    return response.json()

def main(conversation_id):
        headers = create_headers(bearer_token)
        # Replace with conversation ID below (Tweet ID of the root Tweet)
        query_params = {
            'query': 'conversation_id:{}'.format(conversation_id),
            'tweet.fields': 'id,text,author_id,created_at,lang,public_metrics,in_reply_to_user_id',
            'start_time': '2007-01-25T00:00:00Z',
            'max_results': 500}
        json_response = connect_to_endpoint(search_url, headers, query_params)
        return json_response
    

import os
no_exist = []
total = df.shape[0]
so_far = 0
results=[]
for i, j in df[56:].iterrows():
    try:
        t = main(j['id'])
        count = t['meta']['result_count']
        so_far+=count
        if count > 0:
            filename = "comments/{}.json".format(j['id'])
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as outfile:
                json.dump(t, outfile)
            results.append(count)
            print("Count", count)
    except:
        no_exist.append(i)
    finally:
        logger.info("Done {}/{}".format(i,total))
    time.sleep(3.1)