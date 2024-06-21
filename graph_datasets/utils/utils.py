import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GCN(nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GAT(nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.lin = nn.Linear(hidden_channels * heads, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def train(model, optimizer, criterion, loader, device):
    model.train()
    for data in loader:
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            out = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None, data.batch.to(device))
            loss = criterion(out, data.y.to(device))
        else:
            X_batch, y_batch = data
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, criterion, loader, device):
    model.eval()
    correct = 0
    loss_ = 0
    with torch.no_grad():
        for data in loader:
            if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                out = model(data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device) if hasattr(data, 'edge_weight') else None, data.batch.to(device))
                loss = criterion(out, data.y.to(device))
                pred = out.argmax(dim=1)
                correct += int((pred == data.y.to(device)).sum())
            else:
                X_batch, y_batch = data
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                loss = criterion(out, y_batch)
                pred = out.argmax(dim=1)
                correct += int((pred == y_batch).sum())
            loss_ += loss.item()
    return correct / len(loader.dataset), loss_ / len(loader.dataset)

def calculate_edge_weight(edge_index, weight):
    return torch.tensor([weight[i.item(), j.item()] for i, j in zip(edge_index[0], edge_index[1])], dtype=torch.float)
