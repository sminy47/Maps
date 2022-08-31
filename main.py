import pandas as pd
import numpy as np
import requests
longitude=126.94
latitude=37.411
url = 'https://dapi.kakao.com/v2/local/search/keyword.json'

#원하는 검색어을 스타벅스 자리에, 경도는 x에, 위도는 y에, 반경은 10000에, 카테고리는 위의 표를 보고 해당되는 카테고리를 적어주세요
params = {'query' : '스타벅스', 'x' : longitude, 'y' : latitude, 'radius' : 10000, 'category_group_code' : 'CE7'}

## 본인의 카카오 맵 API의 REST API키를 바로 아래 한글로 된 코드를 지우고 입력해주세요
headers = {"Authorization": "KakaoAK d35655e06e78b39688c1ebb6ec01363e"}

total = requests.get(url, params=params, headers=headers).json()['meta']['total_count']

##places는 검색이 잘 되었는지 체크하는 용도로 확인해주시면 됩니다. 다시 말씀드리지만 places에는 45개 데이터가 한계입니다...
## 페이지 수를 늘려도, 한 페이지 안에서 보여줄 수 있는 한계치를 아무리 높혀도 45개 이상 안 보여 줍니다. 저는 페이지 설정은 보시다시피 하지는 않았습니다. 
places = requests.get(url, params=params, headers=headers).json()['documents']

## 원하는 개수는 total변수 안에 있습니다.
print(total)