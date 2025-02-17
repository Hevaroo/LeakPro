"""Implementation of the PLGMI attack."""
import kornia
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from leakpro.attacks.minv_attacks.abstract_minv import AbstractMINV
from leakpro.attacks.utils import losses
from leakpro.attacks.utils.gan_handler import GANHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import MinvResult
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackPLGMI(AbstractMINV):
    """Class that implements the PLGMI attack."""

    def __init__(self: Self, handler: AbstractInputHandler, configs: dict) -> None:
        super().__init__(handler)
        """Initialize the PLG-MI attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """

        logger.info("Configuring PLG-MI attack")
        self._configure_attack(configs)

    def _configure_attack(self: Self, configs: dict) -> None:
        """Configure the attack parameters.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        # TODO: There are some optimizer specific parameters that need to be set here.
        self.num_classes = configs.get("num_classes") # TODO: fail check
        self.batch_size = configs.get("batch_size", 32)
        self.top_n = configs.get("top_n", 10)
        # General parameters
        self.alpha = configs.get("alpha", 0.1)
        self.n_iter = configs.get("n_iter", 1000)
        self.log_interval = configs.get("log_interval", 100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Generator parameters
        self.gen_lr = configs.get("gen_lr", 0.0002) # Learning rate of the generator
        self.gen_beta1 = configs.get("gen_beta1", 0.0) # Beta1 parameter of the generator
        self.gen_beta2 = configs.get("gen_beta2", 0.9) # Beta2 parameter of the generator
        # Discriminator parameters
        self.n_dis = configs.get("n_dis", 5) # Number of discriminator updates per generator update
        self.dis_lr = configs.get("dis_lr", 0.0002)
        self.dis_beta1 = configs.get("dis_beta1", 0.0)
        self.dis_beta2 = configs.get("dis_beta2", 0.9)
        # Paths
        self.pseudo_label_path = configs.get("pseudo_label_path")
        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            # alpha, 0 to inf
            "alpha": (self.alpha, 0, 1000), # 0 to inf
            "n_dis": (self.n_dis, 1, 1000), # 1 to inf
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)


    def description(self:Self) -> dict:
        """Return the description of the attack."""
        title_str = "PLG-MI Attack"
        reference_str = "https://arxiv.org/abs/2302.09814"
        summary_str = "This attack is a model inversion attack that uses the PLG-MI algorithm."
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }


    def top_n_selection(self:Self) -> DataLoader:
        """"Top n selection of pseudo labels."""
        # TODO: This does not scale well. Consider creating a class for the dataloader and implementing the __getitem__ method.
        # TODO: Does this go into modality extension; image?
        logger.info("Performing top-n selection for pseudo labels")
        self.target_model = self.handler.target_model

        self.target_model.eval()
        all_confidences = []
        for images, _ in self.public_dataloader:
            with torch.no_grad():
                outputs = self.target_model(images)
                confidences = F.softmax(outputs, dim=1)
                all_confidences.append(confidences)
        # Concatenate all confidences
        self.confidences = torch.cat(all_confidences)

        logger.info("Retrieved confidences from the target model")
        # Get the pseudo labels
        pseudo_labels = torch.max(self.confidences, dim=1)

        # Empty array of size num_classes to store the pseudo labels
        pseudo_map = [[] for _ in range(self.num_classes)]

        for i, (conf, label) in enumerate(zip(pseudo_labels[0], pseudo_labels[1])):
            pseudo_map[label.item()].append((i, conf.item()))

        # Sort pseudo_map by confidence descending
        for i in range(self.num_classes):
            pseudo_map[i] = sorted(pseudo_map[i], key=lambda x: x[1], reverse=True)

        # keep only top_n entries in each element of pseudo_map
        pseudo_map = [pseudo_map[i][:self.top_n] for i in range(self.num_classes)]

        # Create pseudo dataloader from pseudo_map
        pseudo_data = []
        for i in range(self.num_classes):
            for index, _ in pseudo_map[i]:
                pseudo_data.append((self.public_dataloader.dataset[index][0], i))

        logger.info("Created pseudo dataloader")

        return DataLoader(pseudo_data, batch_size=self.batch_size, shuffle=True)


    def prepare_attack(self:Self) -> None:
        """Prepare the attack."""
        logger.info("Preparing attack")

        self.gan_handler = GANHandler(self.handler)
        # TODO: Change structure of how we load data, handler should do this, not gan_handler
        # Get public dataloader
        self.public_dataloader = self.gan_handler.get_public_data(self.batch_size)

        # Top-n-selection to get pseudo labels
        self.pseudo_loader = self.top_n_selection()

        # Get discriminator
        self.discriminator = self.gan_handler.get_discriminator()

        # Get generator
        self.generator = self.gan_handler.get_generator()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Set Adam optimizer for both generator and discriminator
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr,
                                              betas=(self.gen_beta1, self.gen_beta2))
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.dis_lr,
                                               betas=(self.dis_beta1, self.dis_beta2))

        # Augmentations for generated images. TODO: Move this to a image modality extension
        self.aug_list = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        ).to(self.device)

        # Train the GAN

        if not self.gan_handler.trained_bool:
            logger.info("Training the GAN")

            self.handler.train_gan(pseudo_loader = self.pseudo_loader,
                                        gen = self.generator,
                                        dis = self.discriminator,
                                        gen_criterion = losses.GenLoss(loss_type="hinge", is_relativistic=False),
                                        dis_criterion = losses.DisLoss(loss_type="hinge", is_relativistic=False),
                                        model = self.target_model,
                                        opt_gen = self.gen_optimizer,
                                        opt_dis = self.dis_optimizer,
                                        n_iter = self.n_iter,
                                        n_dis  = self.n_dis,
                                        num_classes = self.num_classes,
                                        device = self.device,
                                        aug_list = self.aug_list,
                                        alpha = self.alpha,
                                        log_interval = self.log_interval,
                                        sample_from_generator = self.gan_handler.sample_from_generator)

            self.gan_handler.trained_bool = True
        else:
            logger.info("GAN already trained, skipping training")

    def run_attack(self:Self) -> MinvResult:
        """Run the attack."""
        # Use trained generator to generate samples and evaluate
        pass
