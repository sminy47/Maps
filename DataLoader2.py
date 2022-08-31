# C:\Users\MinYeong Seo\PycharmProjects\Maps
# https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
import torch
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
import pandas as pd
from torch_geometric.data import HeteroData


#DataSet_카테고리 node 따로(HeteroData)
#지금은 카테고리에서 id를 노드로 받지만
#이후에는 카테고리에서 category_name으로 노드 열로 쭉 세워서 받기(속해있는 관계는 뭘로 나타내지? 엣지?)

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

        building_data = [f for f in self.file_list if f.startswith('building_')]
        building_data.sort()
        self.building_data = building_data    #string, file name

        category_data = [f for f in self.file_list if f.startswith('category_')]
        category_data.sort()
        self.category_data = category_data    #string, file name


    def __len__(self):
        return len(self.building_data)

    def __getitem__(self, index):
        path_dir = self.folder + self.building_data[index]
        building_data = pd.read_csv(path_dir)
        data_id = building_data['id']
        data_x = building_data['y'] #위도
        data_y = building_data['x'] #경도
        c_path_dir = self.folder + self.category_data[index]
        category_data = pd.read_csv(c_path_dir)

        return building_data, path_dir, category_data, c_path_dir


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
            for category in col.split(self.sep):
                x[i, mapping[category]] = 1
                print(category)
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

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


#main
data_dir = 'C:/Users/MinYeong Seo/PycharmProjects/Maps'

if __name__ == "__main__":
    dataset = Dataset(data_dir,train=True)
    data_len =len(dataset)
    #print(data_len)
    it = iter(dataset)

    for i in range(0,data_len):
        #print(i, next(it))
        csv_path_building = dataset.__getitem__(i)[1]
        csv_path_category = dataset.__getitem__(i)[3]
        building_x, building_mapping = load_node_csv(
            csv_path_building, index_col='id', encoders={
                'address_name': SequenceEncoder(),
 #               'category_name': CategroyNameEncoder()
            })
        _, category_mapping = load_node_csv(csv_path_category, index_col='id', encoders={'category_name': CategroyNameEncoder()})  #id가 아니라 카테고리명으로 바꿔야 할 듯
#        print("building_x",len(building_x),building_x)
#        print("building_mapping",len(building_mapping),building_mapping)
#        print(csv_path_building, csv_path_category)

        #Creating Heterogeneous Graphs
        data=HeteroData()
        data['category'].num_nodes = len(category_mapping)
        data['building'].x = building_x

        edge_index, edge_label = load_edge_csv(
            csv_path_category,
            src_index_col='id',
            src_mapping=category_mapping,
            dst_index_col='id',
            dst_mapping=building_mapping)
#            encoders={'rating': IdentityEncoder(dtype=torch.long)},    #나중에 카테고리 이름 인코더

        data['category', 'includes', 'building'].edge_index = edge_index
        data['category', 'includes', 'building'].edge_label = edge_label
        print(edge_index, edge_label)
        print(data)

        node_types, edge_types = data.metadata()
        print("node_types",node_types)
        print(edge_types)

"""
# DataLoader
loader = DataLoader(data, batch_size=32, shuffle=True)
for data in loader:
    print(data)
"""
#건물 분포도 확인용 함수 작성
#각 카테고리간 건물 평균 거리
train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('building', data['building'].x),
)

batch = next(iter(train_loader))
print(batch)
