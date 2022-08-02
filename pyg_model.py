import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, GATConv, DenseGCNConv, GCNConv, GraphConv
import torch
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = GCNConv(165, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(128, 1) 

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)

        return F.sigmoid(x)
