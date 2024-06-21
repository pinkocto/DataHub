import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_umap_data(d):
    umap_data_path = "./data/umap_data/"
    umap_file = f'umap_{d}.csv'
    return pd.read_csv(umap_data_path + umap_file)
