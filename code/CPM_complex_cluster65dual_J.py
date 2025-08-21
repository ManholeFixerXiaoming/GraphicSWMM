import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Global variables
model_loaded = False
net = None
x_combine = None
edge_index = None
edge_attr = None
UP = None

# Define GCN model class
from torch.nn import BatchNorm1d


class GCNNodeEdge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_node, out_channels_edge, rain_length, lstm_hidden=121,
                 point_num=35):
        super(GCNNodeEdge, self).__init__()
        # Define GCN layers
        self.Gconv1 = GCNConv(in_channels, hidden_channels)
        self.Gconvlid = torch.nn.Linear(65 * 3, hidden_channels)
        self.Gconv2 = GCNConv(hidden_channels, 2 * hidden_channels)
        self.Gconv3 = GCNConv(2 * hidden_channels, 1 * hidden_channels)
        self.Gconv4 = GCNConv(1 * hidden_channels, 24)

        # Add 1D Batch Normalization layers
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bnlid = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(2 * hidden_channels)
        self.bn3 = BatchNorm1d(1 * hidden_channels)
        self.bn4 = BatchNorm1d(out_channels_node)

    def forward(self, x_in, edge_index, edge_weight, lid_timeseries, point_num=988):
        edge_weight = edge_weight.squeeze()
        x = self.Gconv1(x_in.squeeze(), edge_index, edge_weight)
        x = self.bn1(x)  # BN layer
        x = F.relu(x)
        lid_timeseries = self.Gconvlid(lid_timeseries.squeeze())
        # lid_timeseries = self.bnlid(lid_timeseries)
        lid_timeseries = F.relu(lid_timeseries)
        x = x + lid_timeseries
        # x = torch.cat((x, lid_timeseries), dim=1)
        mask = (x_in[:, 0] == 0)
        x = x * (~mask).unsqueeze(1)
        x = self.Gconv2(x, edge_index, edge_weight)
        x = self.bn2(x)  # BN layer
        x = F.relu(x)
        x = self.Gconv3(x, edge_index, edge_weight)
        x = self.bn3(x)  # BN layer
        x = F.relu(x)
        x = self.Gconv4(x, edge_index, edge_weight)
        x = self.bn4(x)  # BN layer
        x = F.relu(x)
        sumx = torch.sum(x)

        return x, sumx


# Function to load model and data
def load_model_and_data():
    global model_loaded, net, x_combine, edge_index, edge_attr, UP
    if not model_loaded:
        edges = np.load(r'edge_index_new.npy')
        edges_weight = np.load(r'edge_weight_new.npy')
        x_combine_final = np.load(r'x_new.npy')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        UP = np.load(r'complex_all_area.npy')
        UP = np.repeat(UP, 3, axis=1)  # UP is (1,65), expand to (1,195) as percentages

        # Load node feature matrix
        x_combine = torch.tensor(x_combine_final, dtype=torch.float).to(device)

        # Convert edge information into torch_geometric format (no need to subtract 1 for new clustered edges)
        edge_index = torch.tensor(edges, dtype=torch.long).to(device)
        edge_attr = torch.tensor(edges_weight, dtype=torch.float).to(device)

        net = GCNNodeEdge(in_channels=9, hidden_channels=128 * 2, out_channels_node=24, out_channels_edge=1,
                          rain_length=121)

        net.to('cuda' if torch.cuda.is_available() else 'cpu')

        model_path = r'FC_embed_light.pth'
        net.load_state_dict(torch.load(model_path)['net'])

        model_loaded = True  # Mark model and data as loaded


# Function to run inference
def run_gcn_model(lid_series):
    global net, x_combine, edge_index, edge_attr, UP
    load_model_and_data()  # Ensure model and data are loaded once
    lid_series = lid_series / UP
    lid_series2 = torch.tensor(lid_series, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        outputs2 = net(x_combine, edge_index, edge_attr, lid_series2.unsqueeze(0))
        outputs1, outputs2 = outputs2  # inverse normalization
    return np.array(outputs2.detach().cpu())  # return node outflow
