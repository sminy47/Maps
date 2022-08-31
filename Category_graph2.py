# C:\Users\MinYeong Seo\PycharmProjects\Maps
# https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
c_path = 'C:/Users/MinYeong Seo/PycharmProjects/Maps/train/Entire_category_df.csv'
df = pd.read_csv(c_path)

class Graph:
    def __int__(self, edges, n):
        self.adjList = [[]for _ in range(n)]
        for (src, dest) in edges:
            self.adjList[src].append(dest)


# 중복을 제거 한 모든 node(카테고리) 트리로 나타내기
categoryName_list = list(set(df["category_name"]))  #['여행 > 숙박 > 호텔',~]
#print(categoryName_list)
CL_len = len(categoryName_list)
Entire_start = False
sum_list = []

for i in range(0, CL_len):
    splitResult = categoryName_list[i].split(' > ')
    #print(len(splitResult),splitResult)
    df_p = pd.DataFrame(splitResult)
    for j in range(0, len(splitResult)):
        splitResult[j] = bytes(splitResult[j], 'utf-8')
#     for j in range(0,len(splitResult)):
    sum_list.append(splitResult)

    if Entire_start == False:
        Entire_C_df = df_p
        Entire_start = True
    else:
        Entire_C_df = Entire_C_df.append(df_p)

print(sum_list)
#처음에 그래프 그리고 이 이후부터 연결해서 그리기
#첫 노드가 같은게 이미 저장되어 있는지 확인



sum_torch_tensor = torch.tensor(sum_list)
print(sum_torch_tensor)

#그래프 넣기



