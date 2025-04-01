import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from leakpro import AbstractInputHandler
from leakpro.attacks.utils import gan_losses
from leakpro.schemas import TrainingOutput
from examples.minv.celebA_attributes.utils.CTGAN_extended import CustomCTGAN
import pickle

import pandas as pd

class Mimic_InputHandler(AbstractInputHandler):
    """Class to handle the user input for the CelebA dataset for plgmi attack."""
    
    def __init__(self, configs: dict) -> None:
        super().__init__(configs=configs)
        print("Configurations:", configs)
        
    def get_criterion(self) -> torch.nn.Module:
        """Set the CrossEntropyLoss for the model."""
        return CrossEntropyLoss()

    def get_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Set the optimizer for the model."""
        return optim.SGD(model.parameters())

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: optim.Optimizer,
        epochs: int,
    ) -> dict:
        """Model training procedure."""

        if not epochs:
            raise ValueError("Epochs not found in configurations")

        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)

        for epoch in range(epochs):
            train_loss, train_acc, total_samples = 0.0, 0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Performance metrics
                preds = outputs.argmax(dim=1)
                train_acc += (preds == labels).sum().item()
                train_loss += loss.item()* labels.size(0)
                total_samples += labels.size(0)

        avg_train_loss = train_loss / len(dataloader)
        train_accuracy = train_acc / total_samples  
        model.to("cpu")

        output = {"model": model, "metrics": {"accuracy": train_accuracy, "loss": avg_train_loss}}
        return TrainingOutput(**output)
    
    
    def evaluate(self, dataloader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module) -> dict:
        """Evaluate the model."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()

        test_loss, test_acc, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                test_acc += (preds == labels).sum().item()
                test_loss += loss.item()* labels.size(0)
                total_samples += labels.size(0)
        
        avg_test_loss = test_loss / len(dataloader)
        test_accuracy = test_acc / total_samples
        model.to("cpu")

        return {"accuracy": test_accuracy, "loss": avg_test_loss}

    def train_gan(self,
                    pseudo_loader: DataLoader,
                    gen: torch.nn.Module,
                    dis: torch.nn.Module,
                    gen_criterion: callable,
                    dis_criterion: callable,
                    inv_criterion: callable,
                    target_model: torch.nn.Module,
                    opt_gen: optim.Optimizer,
                    opt_dis: optim.Optimizer,
                    n_iter: int,
                    n_dis: int,
                    device: torch.device,
                    alpha: float,
                    log_interval: int,
                    sample_from_generator: callable
                  ) -> None:
        """Train the CTGAN model. Inspired by CTGAN from https://github.com/sdv-dev/CTGAN.
        
            Args:
                pseudo_loader: DataLoader for the pseudo data.
                gen: Generator model.
                dis: Discriminator model.
                gen_criterion: Generator criterion.
                dis_criterion: Discriminator criterion.
                inv_criterion: Inverted criterion.
                target_model: Target model.
                opt_gen: Generator optimizer.
                opt_dis: Discriminator optimizer.
                n_iter: Number of iterations.
                n_dis: Number of discriminator updates per generator update.
                device: Device to run the training.
                alpha: Alpha value for the invariance loss.
                log_interval: Log interval.
                sample_from_generator: Function to sample from the generator.
        """
        torch.set_default_device(device)
        torch.backends.cudnn.benchmark = True

        target_model.to(device)
        
        ctgan = gen
        
        continuous_col_names = ['length_of_stay', 'num_procedures', 'num_medications', 'BMI',
       'BMI (kg/m2)', 'Height', 'Height (Inches)', 'Weight', 'Weight (Lbs)',
       'eGFR', 'systolic', 'diastolic']
        # Categorical column names, the rest are categorical
        discrete_columns = [col for col in pseudo_loader.dataset.columns if col not in continuous_col_names]
        
        #print(discrete_columns)
        
        #print(pseudo_loader.dataset.head())
        # ctgan takes dataframe or numpy array as input
        ctgan.fit(train_data= pseudo_loader.dataset, 
                    target_model=target_model,
                    num_classes=7011,
                    inv_criterion=inv_criterion,
                    gen_criterion=gen_criterion,
                    dis_criterion=dis_criterion,
                    alpha=alpha,
                    discrete_columns=discrete_columns,
                    use_inv_loss=True,
                    n_iter=n_iter,
                    n_dis=n_dis)
        
        
        ctgan.save("ctgan.pth")
        

        
    
    