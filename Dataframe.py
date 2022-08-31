# C:\Users\MinYeong Seo\PycharmProjects\Maps
# https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import InMemoryDataset
import pandas as pd

#DataSet_카테고리 node에 포함

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if train:
            self.folder = self.data_dir + '/train/'
        else:
            self.folder = self.data_dir + '/test/'
        self.file_list = os.listdir(self.folder)
        file_list_len = len(self.file_list)

        map_data = [f for f in self.file_list if f.startswith('df_')]
        map_data.sort()
        self.map_data = map_data    #string, file name

    def __len__(self):
        return len(self.map_data)

    def __getitem__(self, index):
        path_dir = self.folder + self.map_data[index]
        map_data = pd.read_csv(path_dir)
        data_id = map_data['id']
        data_x = map_data['y'] #위도
        data_y = map_data['x'] #경도
        return map_data, path_dir


#main
data_dir = 'C:/Users/MinYeong Seo/PycharmProjects/Maps'

if __name__ == "__main__":
    dataset = Dataset(data_dir,train=True)
    data_len =len(dataset)
    #print(data_len)
    it = iter(dataset)
    Entire_start = False


    for i in range(0,data_len):
        #print(i, next(it))
        csv_path = dataset.__getitem__(i)[1]
        df = dataset.__getitem__(i)[0]
        dataNum = str(i)

        #건물
        B_df = df[["address_name","distance","id","place_name","road_address_name","x","y"]]
        #print(B_df)
        saveB_path = dataset.folder+ '/building_df'+dataNum+'.csv'
        B_df.to_csv(saveB_path)
        #카테고리
        C_df = df[["category_name","id"]]
        #print(C_df)
        saveC_path = dataset.folder+ '/category_df'+dataNum+'.csv'
        C_df.to_csv(saveC_path)
        #전체카테고리
        if Entire_start==False:
            Entire_C_df = C_df
            Entire_start = True
        else:
            Entire_C_df = Entire_C_df.append(C_df)

    #print(Entire_C_df)
    saveEntireC_path = dataset.folder + '/Entire_category_df' + '.csv'
    Entire_C_df.to_csv(saveEntireC_path)



