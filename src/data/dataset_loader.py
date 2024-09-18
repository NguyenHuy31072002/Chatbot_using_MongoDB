import pandas as pd

def load_dataset(file_path):
    dataset_df = pd.read_csv(file_path)
    return dataset_df
