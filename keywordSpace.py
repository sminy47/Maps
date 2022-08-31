import pandas as pd
import numpy as np
import requests
from folium.plugins import MiniMap
import folium
longitude=127.03684   #경도 x, GetData에서 작성한 것과 같을려면 0.01씩 각각 경도 위도에 더해야 함. 여기는 중심이고 새로운 코드는 왼쪽 좌표이기 때문에
latitude=37.49595 #위도 y
url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
#good
#카페, (한 변이 2.25km인 사각형)
#'radius' : 1125
#'rect': f'{down_x},{down_y},{up_x},{up_y}'
#안양동 32-1
#적도에서 경도 1도의 거리는 약 111.321km인 반면, 위도 60도에서 경도 1도는 55.802km
#지번 사용시 동까지 작성
#0.02씩 사각형
down_x = longitude - 0.01
down_y = latitude - 0.01
up_x = longitude + 0.01
up_y = latitude + 0.01
print(down_x,down_y,up_x,up_y)
#query로 넣을 거 list로 작성해서 추후에 for문을 통해 돌게 하기
params = {'query' : '학원', 'x' : longitude, 'y' : latitude,'rect': f'{down_x},{down_y},{up_x},{up_y}'}  #추후 사각형,,함수내에 띄어쓰기 없애기,,,


headers = {"Authorization": "KakaoAK d35655e06e78b39688c1ebb6ec01363e"}
total = requests.get(url, params=params, headers=headers).json()['meta']['total_count']
#10개씩 끊어서 저장하기
if total >10:
    print()


#places < 45개
places = requests.get(url, params=params, headers=headers).json()['documents']

print("total",total)
print(places)

df = pd.DataFrame.from_records(places)
#df3 = df1.append(df2)
df.to_csv("df_home.csv")
print(df)


