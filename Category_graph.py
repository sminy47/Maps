# C:\Users\MinYeong Seo\PycharmProjects\Maps
# https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd

df = pd.read_csv('C:/Users/MinYeong Seo/PycharmProjects/Maps/train/Entire_category_df.csv')

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

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
    #print(df_p)
#     for j in range(0,len(splitResult)):
    sum_list.append(splitResult)

    if Entire_start == False:
        Entire_C_df = df_p
        Entire_start = True
    else:
        Entire_C_df = Entire_C_df.append(df_p)
#처음에 그래프 그리고 이 이후부터 연결해서 그리기
#첫 노드가 같은게 이미 저장되어 있는지 확인
print(sum_list)
#sum_torch_tensor = torch.tensor(sum_list)
#print(sum_torch_tensor)
#edge_index =
#x = torch.tensor(sum_torch_tensor, dtype=str)
#data = Data(x=x)

#print(data)



