import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
import torchmetrics.functional as MF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")
from dgl_model import SAGE
from read_data import process_ellipitc
import dgl
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import argparse


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        return MF.accuracy(pred, label)

def train(args, device, g, train_idx, val_idx, model):
    # create sampler & dataloader
    sampler = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))

if __name__ == '__main__':

    df_features = pd.read_csv('raw/elliptic_txs_features.csv', header=None)
    df_edges = pd.read_csv("raw/elliptic_txs_edgelist.csv")
    df_classes =  pd.read_csv("raw/elliptic_txs_classes.csv")

    node_features, classified_idx, edge_index, weights, labels, y_train = process_ellipitc(df_features, df_edges, df_classes)
# converting data to PyGeometric graph data format

    X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(node_features[classified_idx], y_train, classified_idx, test_size=0.15, random_state=42, stratify=y_train)
    train_idx = torch.LongTensor(train_idx)
    valid_idx = torch.LongTensor(valid_idx)
    print (X_train.size(), X_valid.size())
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    g = dgl.graph((edge_index[0],edge_index[1]))
    g.ndata['feat'] = node_features.to(torch.float)
    g.ndata['label'] = torch.LongTensor(labels)
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    train_idx = train_idx.to('cuda' if args.mode == 'puregpu' else 'cpu')
    valid_idx = valid_idx.to('cuda' if args.mode == 'puregpu' else 'cpu')

    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = 2
    model = SAGE(in_size, 256, out_size).to(device)
    # model training
    print('Training...')
    train(args, device, g, train_idx, valid_idx, model)


