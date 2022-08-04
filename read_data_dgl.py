import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
import dgl
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, GATConv, DenseGCNConv, GCNConv, GraphConv
from torch_geometric.data import Data, DataLoader
import warnings
warnings.filterwarnings("ignore")

import cudf


def process_ellipitc(df_features, df_edges, df_classes):

    #df_features = pd.read_csv('raw/elliptic_txs_features.csv', header=None)
    #df_edges = pd.read_csv("raw/elliptic_txs_edgelist.csv")
    #df_classes =  pd.read_csv("raw/elliptic_txs_classes.csv")
    df_classes['class'] = df_classes['class'].map({'unknown': 2, '1':1, '2':0})

    # merging dataframes
    df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
    df_merge = df_merge.sort_values(0).reset_index(drop=True)
    classified = df_merge.loc[df_merge['class'].loc[df_merge['class']!=2].index].drop('txId', axis=1)
    unclassified = df_merge.loc[df_merge['class'].loc[df_merge['class']==2].index].drop('txId', axis=1)

    # storing classified unclassified nodes seperatly for training and testing purpose
    classified_edges = df_edges.loc[df_edges['txId1'].isin(classified[0]) & df_edges['txId2'].isin(classified[0])]
    unclassifed_edges = df_edges.loc[df_edges['txId1'].isin(unclassified[0]) | df_edges['txId2'].isin(unclassified[0])]
    del df_features, df_classes


    # all nodes in data
    nodes = df_merge[0].values
    map_id = {j:i for i,j in enumerate(nodes)} # mapping nodes to indexes

    edges = df_edges.copy()
    edges.txId1 = edges.txId1.map(map_id)
    edges.txId2 = edges.txId2.map(map_id)
    edges = edges.astype(int)

    edge_index = np.array(edges.values).T

    # for undirected graph
    # edge_index_ = np.array([edge_index[1,:], edge_index[0, :]])
    # edge_index = np.concatenate((edge_index, edge_index_), axis=1)

    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.double)
    print(edge_index.shape)


    # maping txIds to corresponding indexes, to pass node features to the model
    node_features = df_merge.drop(['txId'], axis=1).copy()
    node_features[0] = node_features[0].map(map_id)
    classified_idx = node_features['class'].loc[node_features['class']!=2].index
    unclassified_idx = node_features['class'].loc[node_features['class']==2].index
    # replace unkown class with 0, to avoid having 3 classes, this data/labels never used in training
    node_features['class'] = node_features['class'].replace(2, 0)


    labels = node_features['class'].values
    node_features = torch.tensor(np.array(node_features.drop([0, 'class', 1], axis=1).values, dtype=np.double), dtype=torch.double)

    y_train = labels[classified_idx]
    return node_features, classified_idx, edge_index, weights, labels, y_train


def convert_to_column_major(t):
    return t.t().contiguous().t()
   

if __name__ == '__main__':
    edge_df = cudf.from_dlpack(to_dlpack(convert_to_column_major(edge_index.t())))
    edge_df['edge_id'] = cp.arange(0,len(edge_df))
    edge_df.columns = ['src','dst','edge_id']

    node_feat_df = cudf.from_dlpack(to_dlpack(convert_to_column_major(node_features))).astype(cp.float32)
    node_feat_df['node_id'] = cp.arange(0,len(node_feat_df))

    node_label_df = cudf.DataFrame({'label':labels})
    node_label_df['node_id'] =  cp.arange(0,len(node_label_df))
