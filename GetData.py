import requests
import pandas as pd
import numpy as np
from folium.plugins import MiniMap
import folium
import collections

# 시작 x 좌표 및 증가값
keyword = '학원'
start_x = 127.03684 #사각형 왼쪽아래
start_y = 37.49595 #사각형 왼쪽아래
next_x = 0.01
next_y = 0.01
num_x = 2
num_y = 2
# 1. 카테고리 리스트 전부 검색
keywordList=['학원','음식점','주거시설','부동산']

# 2. 좌표 이동

##카카오 API
def whole_region(keyword, start_x, start_y, end_x, end_y):
    #print("pos: ",start_x,start_y,end_x,end_y)
    page_num = 1
    # 데이터가 담길 리스트
    all_data_list = []

    while (1):
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
        params = {'query': keyword, 'page': page_num,
                  'rect': f'{start_x},{start_y},{end_x},{end_y}'}
        headers = {"Authorization": "KakaoAK d35655e06e78b39688c1ebb6ec01363e"}
        resp = requests.get(url, params=params, headers=headers)

        search_count = resp.json()['meta']['total_count']
        print('총 개수', search_count)

        if search_count > 45:
            print('좌표 4등분')
            dividing_x = (start_x + end_x) / 2
            dividing_y = (start_y + end_y) / 2
            ## 4등분 중 왼쪽 아래
            all_data_list.extend(whole_region(keyword, start_x, start_y, dividing_x, dividing_y))
            ## 4등분 중 오른쪽 아래
            all_data_list.extend(whole_region(keyword, dividing_x, start_y, end_x, dividing_y))
            ## 4등분 중 왼쪽 위
            all_data_list.extend(whole_region(keyword, start_x, dividing_y, dividing_x, end_y))
            ## 4등분 중 오른쪽 위
            all_data_list.extend(whole_region(keyword, dividing_x, dividing_y, end_x, end_y))
            return all_data_list

        else:
            if resp.json()['meta']['is_end']:
                all_data_list.extend(requests.get(url, params=params, headers=headers).json()['documents'])
                #all_data_list.extend(resp.json()['documents'])
                return all_data_list
            # 아니면 다음 페이지로 넘어가서 데이터 저장
            else:
                print('다음페이지')
                page_num += 1
                #all_data_list.extend(resp.json()['documents'])
                all_data_list.extend(requests.get(url, params=params, headers=headers).json()['documents'])

def overlapped_data(keyword, start_x, start_y, next_x, next_y, num_x, num_y):
    # 최종 데이터가 담길 리스트
    overlapped_result = []

    # 지도를 사각형으로 나누면서 데이터 받아옴
    for i in range(1, num_x + 1):  ## 1,10
        end_x = start_x + next_x
        initial_start_y = start_y
        for j in range(1, num_y + 1):  ## 1,6
            end_y = initial_start_y + next_y
            each_result = whole_region(keyword, start_x, initial_start_y, end_x, end_y)
            overlapped_result.extend(each_result)
            initial_start_y = end_y
        start_x = end_x

    return overlapped_result


def make_map(dfs):
    # 지도 생성하기
    m = folium.Map(location=[start_y, start_x],  # 기준좌표
                   zoom_start=12)

    # 미니맵 추가
    minimap = MiniMap()
    m.add_child(minimap)

    # 마커
    for i in range(len(dfs)):
        folium.Marker([df['y'][i], df['x'][i]],
                      tooltip=dfs['place_name'][i],
                      popup=dfs['place_url'][i],
                      ).add_to(m)
    return m


#main
overlapped_result = overlapped_data(keyword, start_x, start_y, next_x, next_y, num_x, num_y)

# 최종 데이터가 담긴 리스트 중복값 제거
results = list(map(dict, collections.OrderedDict.fromkeys(tuple(sorted(d.items())) for d in overlapped_result)))
df = pd.DataFrame.from_records(results)

print('total_reuslt_number = ', len(df))
df.to_csv("df_new.csv")
mm = make_map(df)
mm.save('map.html')