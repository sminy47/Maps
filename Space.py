import pandas as pd
import numpy as np
import requests

longitude=126.94
latitude=37.411
url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
down_x = round(longitude - (1.125/111.321),5)
down_y = round(latitude - (1.125/55.802),5)
up_x = round(longitude + (1.125/111.321),5)
up_y = round(latitude + (1.125/55.802),5)
params = {'query' : '','x' : longitude, 'y' : latitude,'rect': (down_x,down_y,up_x,up_y), 'category_group_code' : 'CE7'}

headers = {"Authorization": "KakaoAK d35655e06e78b39688c1ebb6ec01363e"}

total = requests.get(url, params=params, headers=headers).json()['meta']['total_count']

##places < 45개
places = requests.get(url, params=params, headers=headers).json()['documents']

print(total)
print(places)

df = pd.DataFrame.from_records(places)
df.to_csv("df.csv", mode='w')
print(df)