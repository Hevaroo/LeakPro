from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CelebATabularDataset(Dataset):
    """Dataset class for the CelebA attributes dataset."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features.iloc[idx], self.labels.iloc[idx]

    @classmethod
    def from_celebA(cls, path):
        # read data from plk file
        df = pd.read_pickle(path)
        features = df.drop('identity', axis=1)
        labels = df['identity']

        return cls(features, labels)

def get_celebA_test_train_loader(train_config):
    """Get the train and test dataloaders for the CelebA dataset."""
    train_fraction = train_config["data"]["f_train"]
    test_fraction = train_config["data"]["f_test"]
    batch_size = train_config["train"]["batch_size"]
    data_dir =  train_config["data"]["data_dir"] + "/private_df.pkl"

    # Load the data
    private_dataset = CelebATabularDataset.from_celebA(data_dir)

    dataset_size = len(private_dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)

    # Split the data into train and test sets stratified by the labels
    train_indices, test_indices = train_test_split(range(dataset_size), test_size=test_size, train_size=train_size, stratify=private_dataset.labels)

    train_subset = torch.utils.data.Subset(private_dataset, train_indices)
    test_subset = torch.utils.data.Subset(private_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_celebA_public_loader(train_config):
    """Get the public CelebA dataset."""
    batch_size = train_config["train"]["batch_size"]
    data_dir =  train_config["data"]["data_dir"] + "/public_df.pkl"

    # Load the data
    public_dataset = CelebATabularDataset.from_celebA(data_dir)

    return DataLoader(public_dataset, batch_size=batch_size, shuffle=False)
