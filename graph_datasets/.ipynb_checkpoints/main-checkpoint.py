import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from utils.utils import MLP, GCN, GAT, train, test, calculate_edge_weight
from utils.load import load_data, load_umap_data
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('./data/cleaned_level7.csv')
    y = pd.read_csv('./data/y.csv').values.ravel()
    y_train, y_test = y[:170], y[170:]
    umap_w = None
    if args.use_edge_weight:
        umap_w = pd.read_csv(args.edge_weight_path).values

    if args.model_type.lower() == 'mlp':
        umap_df = load_umap_data(args.umap_dim)
        X = umap_df.values
        X_train, X_test = X[:170], X[170:]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        input_dim = X_train.shape[1]
        output_dim = len(set(y))
        model = MLP(input_dim, args.hidden_dim, output_dim).to(device)

    elif args.model_type.lower() == 'gcn':
        from torch_geometric.data import DataLoader
        dataset_name = f'CLASS7_U{args.umap_dim}{args.method}'
        dataset = MCRDataset(root='.', name=dataset_name)
        dataset = dataset.shuffle()

        graphs = []
        for data in dataset:
            if args.use_edge_weight:
                edge_weight = calculate_edge_weight(data.edge_index, umap_w)
                data.edge_weight = edge_weight
            graphs.append(data)

        train_loader = DataLoader(graphs[:170], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(graphs[170:], batch_size=args.batch_size, shuffle=False)

        model = GCN(hidden_channels=args.hidden_dim, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)

    elif args.model_type.lower() == 'gat':
        from torch_geometric.data import DataLoader
        dataset_name = f'CLASS7_U{args.umap_dim}{args.method}'
        dataset = MCRDataset(root='.', name=dataset_name)
        dataset = dataset.shuffle()

        train_dataset = dataset[:170]
        test_dataset = dataset[170:]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = GAT(hidden_channels=args.hidden_dim, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes, heads=args.heads).to(device)

    else:
        raise ValueError("Unknown model type")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, criterion, train_loader, device)
        train_acc, train_loss = test(model, criterion, train_loader, device)
        test_acc, test_loss = test(model, criterion, test_loader, device)
        print(f'Epoch {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate GCN, GAT, or MLP models")
    parser.add_argument('--model_type', type=str, choices=['MLP', 'GCN', 'GAT'], required=True)
    parser.add_argument('--umap_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--method', type=str, default='PC')
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--use_edge_weight', action='store_true')
    parser.add_argument('--edge_weight_path', type=str)
    args = parser.parse_args()
    main(args)
