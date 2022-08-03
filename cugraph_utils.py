import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

# Example based on
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py
## cugraph imports
import cugraph
import cudf
import cupy
import time

# util to create property graph from ns
def add_ndata(gs, graph):
    if len(graph.ntypes) != 1:
        raise "graph.ntypes!=1 not currently supported"

    for key in graph.ndata.keys():
        # TODO: Change to_dlpack
        # tensor = tensor.to('cpu')
        # ar = cupy.from_dlpack(F.zerocopy_to_dlpack(tensor))
        ar = graph.ndata[key].cpu().detach().numpy()
        cudf_data_obj = cudf.DataFrame(ar)
        # handle 1d tensors
        if isinstance(cudf_data_obj, cudf.Series):
            df = cudf_data_obj.to_frame()
        else:
            df = cudf_data_obj

        df.columns = [f"{key}_{i}" for i in range(len(df.columns))]
        node_ids = dgl.backend.zerocopy_to_dlpack(graph.nodes())
        node_ser = cudf.from_dlpack(node_ids)
        df["node_id"] = node_ser

        gs.add_node_data(df, "node_id", key, graph.ntypes[0])
    return gs


def add_edata(gs, graph):
    src_t, dst_t = graph.edges()
    edge_ids = graph.edge_ids(src_t, dst_t)
    df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(dgl.backend.zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(dgl.backend.zerocopy_to_dlpack(dst_t)),
            "edge_id": cudf.from_dlpack(dgl.backend.zerocopy_to_dlpack(edge_ids)),
        }
    )
    gs.add_edge_data(df, ["src", "dst"], "edge_id")
    return gs


def create_cugraph_store(graph):
    # create pg from cugraph graph
    pg = cugraph.experimental.PropertyGraph()
    # create gs from pg
    gs = dgl.contrib.cugraph.CuGraphStorage(pg)
    add_edata(gs, graph)
    add_ndata(gs, graph)
    return gs


