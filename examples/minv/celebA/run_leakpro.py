import os
import sys
import yaml


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(project_root)


train = False
if train:
    from examples.minv.celebA.utils.celebA_data import get_celebA_train_test_loader

    # Load the config.yaml file
    with open('train_config.yaml', 'r') as file:
        train_config = yaml.safe_load(file)

    # Generate the dataset and dataloaders
    path = os.path.join(os.getcwd(), train_config["data"]["data_dir"])

    #print(train_config)

    train_loader, test_loader = get_celebA_train_test_loader(train_config)

    num_classes = train_loader.dataset.dataset.get_classes()
    # ResNet152 model
    from examples.minv.celebA.utils.resnet152_model import ResNet152
    from examples.minv.celebA.utils.resnet152_model import create_trained_model_and_metadata
    # Get number of classes from the train_loader
    num_classes = train_loader.dataset.dataset.get_classes()
    print(num_classes)

    import torch
    # Create the model
    model = ResNet152(num_classes=num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    # Load the model
    train_acc, train_loss, test_acc, test_loss = create_trained_model_and_metadata(model,train_loader,test_loader, train_config)

    print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

from leakpro import LeakPro
from examples.minv.celebA.celebA_plgmi_handler import CelebA_InputHandler
config_path = "audit.yaml"


# Initialize the LeakPro object
leakpro = LeakPro(CelebA_InputHandler, config_path)

# Run the audit
results = leakpro.run_audit(return_results=True)
