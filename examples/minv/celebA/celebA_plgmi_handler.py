import torch
from torch import cuda, device, optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from leakpro import AbstractInputHandler
from leakpro.attacks.utils import losses

class CelebA_InputHandler(AbstractInputHandler):
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
            train_loss, train_acc = 0.0, 0
            model.train()
            for inputs, labels in tqdm(dataloader, desc="Training Progress"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Performance metrics
                preds = outputs.argmax(dim=1)
                train_acc += (preds == labels).sum().item()
                train_loss += loss.item()

        model.to("cpu")

        return {"model": model, "metrics": {"accuracy": train_acc / len(dataloader.dataset), "loss": train_loss}}
    
    
    def evaluate(self, dataloader: DataLoader, model: torch.nn.Module, criterion: torch.nn.Module) -> dict:
        """Evaluate the model."""
        gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
        model.to(gpu_or_cpu)
        model.eval()

        test_loss, test_acc = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(gpu_or_cpu), labels.to(gpu_or_cpu)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                test_acc += (preds == labels).sum().item()
                test_loss += loss.item()

        model.to("cpu")

        return {"accuracy": test_acc / len(dataloader.dataset), "loss": test_loss}

    def train_gan(self,
                    pseudo_loader: DataLoader,
                    gen: torch.nn.Module,
                    dis: torch.nn.Module,
                    gen_criterion: torch.nn.Module,
                    dis_criterion: torch.nn.Module,
                    model: torch.nn.Module,
                    opt_gen: optim.Optimizer,
                    opt_dis: optim.Optimizer,
                    n_iter: int,
                    n_dis: int,
                    num_classes: int,
                    device: torch.device,
                    aug_list: list,
                    alpha: float,
                    log_interval: int,
                    sample_from_generator: callable
                  ) -> None:
        """Train the GAN model. Copied from https://github.com/LetheSec/PLG-MI-Attack."""
        torch.set_default_device(device)
        torch.backends.cudnn.benchmark = True
        gen_losses = []
        dis_losses = []
        inv_losses = []

        # Training loop
        for i in range(n_iter):
            _l_g = .0
            cumulative_inv_loss = 0.
            cumulative_loss_dis = .0

            cumulative_target_acc = .0
            target_correct = 0
            count = 0
            for j in range(n_dis):
                if j == 0:
                    fake, fake_labels, _ = sample_from_generator(gen, num_classes, 128, device, gen.dim_z)
                    print("Sampled from generator")
                    fake_aug = aug_list(fake)
                    print("Augmented fake images")
                    dis_fake = dis(fake_aug, fake_labels)
                    print("Discriminator fake images")
                    inv_loss = losses.max_margin_loss(model(fake_aug), fake_labels)
                    print("Calculated max margin loss")
                    
                    inv_losses.append(inv_loss.item())
                    dis_real = None

                    loss_gen = gen_criterion(dis_fake, dis_real)
                    gen_losses.append(loss_gen.item())
                    loss_all = loss_gen + inv_loss*alpha

                    gen.zero_grad()
                    loss_all.backward()
                    opt_gen.step()
                    _l_g += loss_gen.item()
                    cumulative_inv_loss += inv_loss.item()

                fake, fake_labels, _ = sample_from_generator(gen, num_classes, 128, device, gen.dim_z)

                real, real_labels = next(iter(pseudo_loader))
                real, real_labels = real.to(device), real_labels.to(device)

                dis_fake = dis(fake, fake_labels)
                dis_real = dis(real, real_labels)

                loss_dis = dis_criterion(dis_fake, dis_real)
            
                dis.zero_grad()
            
                loss_dis.backward()
                opt_dis.step()

                cumulative_loss_dis += loss_dis.item()
                dis_losses.append(cumulative_loss_dis/n_dis)
                
                with torch.no_grad():
                    count += fake.shape[0]
                    T_logits = model(fake)
                    T_preds = T_logits.max(1, keepdim=True)[1]
                    target_correct += T_preds.eq(fake_labels.view_as(T_preds)).sum().item()
                    cumulative_target_acc += round(target_correct / count, 4)

            if i % log_interval == 0:
                print(
                        'iteration: {:05d}/{:05d}, loss gen: {:05f}, loss dis {:05f}, inv loss {:05f}, target acc {:04f}'.format(
                            i, n_iter, _l_g, cumulative_loss_dis, cumulative_inv_loss,
                            cumulative_target_acc, ))
                
                if cumulative_target_acc > 0.2:
                    break
