import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Load the dataframes from pickle files
private_df = pd.read_pickle('private_df.pkl')
public_df = pd.read_pickle('public_df.pkl')

class CelebATabularDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        # Extract features and labels from the sample
        # Assuming 'identity' is the label and the rest are features
        features = sample.drop('identity').values
        label = sample['identity']
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Create datasets
private_dataset = CelebATabularDataset(private_df)
public_dataset = CelebATabularDataset(public_df)

# Create dataloaders
private_dataloader = DataLoader(private_dataset, batch_size=32, shuffle=True)
public_dataloader = DataLoader(public_dataset, batch_size=32, shuffle=True)

# Example of iterating through the dataloader
for features, labels in private_dataloader:
    print(features, labels)
    break