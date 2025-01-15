"""Hyperparameter tuning with optuna on evaluating."""

from cifar import get_cifar10_loader

from leakpro.attacks.gia_attacks.huang import Huang, HuangConfig
from leakpro.fl_utils.gia_train import train
from leakpro.optuna import optuna_optimal_hyperparameters
from model import ResNet, PreActBlock

if __name__ == "__main__":
    # Pre activation required for this attack to give decent results
    base_model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=10)
    cifar10_loader, mean, std = get_cifar10_loader(num_images=16, batch_size=16, num_workers=2)

    # Run Optuna optimization with Huang
    attack_object = Huang(base_model, cifar10_loader, train, mean, std, HuangConfig())
    optuna_optimal_hyperparameters(attack_object=attack_object)