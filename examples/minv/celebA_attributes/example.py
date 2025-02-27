import os
import sys
import yaml

from examples.minv.celebA_attributes.utils import get_celebA_train_test_loader, get_celebA_public_loader


# Path to the dataset zip file
data_folder = "./data"


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)

# Load the config.yaml file
with open('train_config.yaml', 'r') as file:
    train_config = yaml.safe_load(file)

# Generate the dataset and dataloaders
path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])

#print(train_config)

train_loader, test_loader = get_celebA_train_test_loader(train_config)

public_loader = get_celebA_public_loader(train_config)