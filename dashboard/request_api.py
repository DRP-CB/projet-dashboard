import pandas as pd
import json
import requests




#data = pd.read_csv('processedData.csv', index_col="SK_ID_CURR")


#data.index[0:10]

url = 'http://localhost:5000/pred'

def send_request(sk_id_curr,url,data):
    payload =  json.dumps(dict(data.loc[sk_id_curr,:]))    
    return requests.post(url,json=payload).json()


#send_request(100004,url,data)



