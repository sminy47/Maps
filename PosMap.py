#x,y(경도, 위도)에 따른 node  위치 찍어보기
from pyproj import Transformer
import pyproj
import numpy as np
import pandas  as pd
#from shapely.geometry import Point as point
#import geopandas as gpd
import folium

#xlsx
bike = pd.read_excel('df_cafe.xlsx') # 해당 엑셀 파일 불러오기
print(bike)
bike_id = bike['id']
bike_x = bike['y'] #위도
bike_y = bike['x'] #경도

# 지도의 중심을 지정하기 위해 위도와 경도의 평균 구하기
lat = bike['y'].mean()
long = bike['x'].mean()

# 지도 띄우기
m = folium.Map([lat, long], zoom_start=9)

coords = []
for i in range(len(bike) - 1):
    x = bike_x[i]
    y = bike_y[i]
    coords.append([x, y])

for i in range(len(coords)):
    folium.Circle(
        location=coords[i],
        radius=50,
        color='#000000',
        fill='crimson',
    ).add_to(m)

m.save('map.html')