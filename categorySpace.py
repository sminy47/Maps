import pandas as pd
import numpy as np
import requests

longitude=126.917
latitude=37.415
url = 'https://dapi.kakao.com/v2/local/search/category.json'

#카페, 2500(2.5km, 1250 반지름 반경)
params = {'x' : longitude, 'y' : latitude, 'radius' : 20000, 'category_group_code' : 'AD5'}

headers = {"Authorization": "KakaoAK d35655e06e78b39688c1ebb6ec01363e"}

total = requests.get(url, params=params, headers=headers).json()['meta']['total_count']

##places < 45개
places = requests.get(url, params=params, headers=headers).json()['documents']

print(total)
print(places)

df = pd.DataFrame.from_records(places)
df.to_csv("df_home2.csv")
print(df)