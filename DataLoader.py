# C:\Users\MinYeong Seo\PycharmProjects\Maps
# https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.data import InMemoryDataset
#from torch_geometric.loader import
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

        map_data = [f for f in self.file_list if f.startswith('building_')]
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


class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

#category_name
class CategroyNameEncoder(object):
    def __init__(self, sep=' > '):
        self.sep = sep

    def __call__(self, df):
        categories= set(g for col in df.values for g in col.split(self.sep))
        mapping = {category: i for i, category in enumerate(categories)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


#Loading Graphs from CSV
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    return x, mapping

#main
data_dir = 'C:/Users/MinYeong Seo/PycharmProjects/Maps'

if __name__ == "__main__":
    dataset = Dataset(data_dir,train=True)
    data_len =len(dataset)
    #print(data_len)
    it = iter(dataset)


    for i in range(0,data_len):
        #print(i, next(it))
        csv_path = dataset.__getitem__(i)[1]
        buliding_x, buliding_mapping = load_node_csv(
            csv_path, index_col='id', encoders={
                'address_name': SequenceEncoder(),
                'category_name': CategroyNameEncoder()
            })
        print("buliding_x",len(buliding_x),buliding_x)
        print("buliding_mapping",len(buliding_mapping),buliding_mapping)




# Loading Graphs from csv

# DataLoader