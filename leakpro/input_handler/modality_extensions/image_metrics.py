"""ImageMetrics class for computing image metrics."""
import os

import joblib
import numpy as np
import torch
from pydantic import BaseModel
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision import models, transforms

from leakpro.attacks.utils.generator_handler import GeneratorHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.input_handler.user_imports import get_class_from_module, import_module_from_file
from leakpro.schemas import MIAMetaDataSchema
from leakpro.utils.logger import logger


class ImageMetrics:
    """Class for computing image metrics."""

    def __init__(self,
                 handler: MINVHandler,
                 generator_handler: GeneratorHandler,
                 configs: dict,
                 labels: torch.tensor = None,
                 z: torch.tensor = None,) -> None:
        """Initialize the ImageMetrics class."""
        self.handler = handler
        self.generator_handler = generator_handler
        self.generator = self.generator_handler.get_generator()

        self.target_model = self.handler.target_model
        self.labels = labels
        self.z = z
        logger.info("Configuring ImageMetrics")
        self._configure_metrics(configs)
        self.test_dict = {
            "accuracy": self.compute_accuracy,
            "fid" : self.compute_fid,
            "knn": self.compute_knn_dist,
            "save_images": self.save_images,
        }
        logger.info(configs)
        self.results = {}

        self.load_evaluation_model()

        # TODO: This loading functionality should not be in generator_handler
        self.private_dataloader = self.handler.get_private_dataloader(self.batch_size)
        # Compute desired metrics from configs
        self.metric_scheduler()

    def _configure_metrics(self, configs: dict) -> None:
        """Configure the metrics parameters.

        Args:
        ----
            configs (dict): Configuration parameters for the metrics.

        """
        self.configs = configs
        self.batch_size = configs.batch_size
        self.num_class_samples = configs.num_class_samples
        self.num_audited_classes = configs.num_audited_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_evaluation_model(self) -> None:
        """Load the evaluation model."""
        model_class = self.configs.eval_model.model_class
        if model_class is None:
            raise ValueError("model_class not found in configs.")

        module_path=self.configs.eval_model.module_path
        if module_path is None:
            raise ValueError("module_path not found in configs.")

        try:
            eval_module = import_module_from_file(module_path)
            self.eval_model_blueprint = get_class_from_module(eval_module, model_class)
            logger.info(f"Eval model blueprint created from {model_class} in {module_path}.")
        except Exception as e:
            raise ValueError(f"Failed to create the eval model blueprint from {model_class} in {module_path}") from e

        """Get the eval model metadata from the trained model metadata file."""
        model_path = self.configs.eval_model.eval_folder
        if model_path is None:
            raise ValueError("Trained model metadata path not found in configs.")
        try:
            self.eval_model_metadata_path = f"{model_path}/model_metadata.pkl"
            with open(self.eval_model_metadata_path, "rb") as f:
                eval_model_metadata = joblib.load(f)

                # check if the metadata is a schema or a dict, initate a schema
                if not isinstance(eval_model_metadata, BaseModel):
                    self.eval_model_metadata = MIAMetaDataSchema(**eval_model_metadata)
                else:
                    self.eval_model_metadata = eval_model_metadata
            logger.info(f"Loaded eval model metadata from {self.eval_model_metadata_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the eval model metadata at {self.eval_model_metadata_path}") from e

        """Get the trained eval model."""
        if model_path is None:
            raise ValueError("Trained model path not found in configs.")
        self.model_path = f"{model_path}/target_model.pkl"
        init_params = self.eval_model_metadata.init_params
        try:
            with open(self.model_path, "rb") as f:
                self.evaluation_model = self.eval_model_blueprint(**init_params)
                self.evaluation_model.load_state_dict(torch.load(f))
            logger.info(f"Loaded eval model from {model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the trained eval model at {model_path}") from e


    def metric_scheduler(self) -> None:
        """Schedule the metrics to be computed."""
        tests = self.configs.metrics
        # If tests empty, return
        if not tests:
            logger.warning("No tests specified in the config.")
            return

        for test in tests:
            if test in self.test_dict:
                self.test_dict[test]()
            else:
                logger.warning(f"Test {test} not found in the test dictionary.")

    def compute_accuracy(self) -> None:
        """Compute accuracy for generated samples.

        We generate samples for each pair of label and z, and compute the accuracy of the evaluation model on these samples.
        """

        logger.info("Computing accuracy for generated samples.")
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)
        self.generator.eval()
        self.generator.to(self.device)
        correct_predictions = []
        for label_i, z_i in zip(self.labels, self.z):
            # generate samples for each pair of label and z
            generated_sample = self.generator_handler.sample_from_generator(batch_size=self.num_class_samples,
                                                                            label=label_i,
                                                                            z=z_i)
            for j in range(self.num_class_samples):
                with torch.no_grad():
                    output = self.evaluation_model(generated_sample[j])
                    prediction = torch.argmax(output, dim=1)
                    correct_predictions.append(prediction == label_i)
        correct_predictions = torch.cat(correct_predictions).float()
        self.accuracy = correct_predictions.mean()
        self.accuracy_std = correct_predictions.std() / torch.sqrt(torch.tensor(len(correct_predictions), dtype=torch.float))
        logger.info(f"Accuracy: {self.accuracy.item()}")

        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()


    def compute_fid(self) -> None:
        """Compute the Frechet Inception Distance (FID) between real and generated images.

        Reference Heusel et al. 2017 https://arxiv.org/abs/1706.08500.
        """

        logger.info("Computing FID between real and generated samples.")

        # Load InceptionV3 model for feature extraction
        inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        inception_model.fc = torch.nn.Identity()  # Remove final classification layer
        inception_model.eval()
        inception_model.to(self.device)

        # Image transformation for InceptionV3 input
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Extract features from real and generated images
        real_features = self.get_features(self.private_dataloader, inception_model, transform).detach().cpu().numpy()
        fake_features = self.get_generated_features(inception_model, transform).detach().cpu().numpy()

        # Compute mean and covariance of features
        mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

        # Calculate FID score
        diff = mu_real - mu_fake
        covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)

        # Store results
        self.results["fid"] = fid_score.item()
        logger.info(f"FID: {fid_score}")

    def compute_knn_dist(self) -> None:
        """Compute the k-NN distance."""

        logger.info("Computing k-NN distance for generated samples.")

        # Image transformation for evaluation model input
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Extract features from private data loader
        logger.info("Extracting features from private data loader.")
        private_features = self.get_features(self.private_dataloader, self.evaluation_model, transform)
        private_targets = np.concatenate([targets.numpy() for _, targets in self.private_dataloader])

        # Extract features from generated images
        logger.info("Extracting features from generated images.")
        generated_features = self.get_generated_features(self.evaluation_model, transform)
        generated_targets = torch.tensor([label for label in self.labels for _ in range(self.num_class_samples)]).to(self.device)

        # Calculate k-NN distance
        def calc_knn(feat, gen_label, true_feat, true_label):  # noqa: ANN001, ANN202
            """Calculate the k-NN distance between generated and true features using vectorized operations."""
            gen_label = gen_label.cpu().long()
            feat = feat.cpu()
            true_feat = true_feat.cpu()
            true_label = true_label.cpu().long()

            knn_dist = 0

            for label in gen_label:
                gen_mask = (gen_label == label).nonzero(as_tuple=True)[0]
                true_mask = (true_label == label).nonzero(as_tuple=True)[0]

                # Filter features by label
                gen_feat = feat[gen_mask]
                true_feat_label = true_feat[true_mask]

                # Skip if no features for the label
                if gen_feat.size(0) == 0 or true_feat_label.size(0) == 0:
                    continue

                # Calculate pairwise distances between generated and true features
                dists = torch.cdist(gen_feat, true_feat_label, p=2)

                # Find the minimum distance for each generated feature
                min_dists, _ = dists.min(dim=1)
                knn_dist += min_dists.sum().item()

            return knn_dist / feat.size(0)

        logger.info("Calculating k-NN distance.")
        knn_dist = calc_knn(generated_features, generated_targets,
                            torch.tensor(private_features),
                            torch.tensor(private_targets))

        # Store results
        self.results["knn_dist"] = knn_dist
        logger.info(f"k-NN Distance: {knn_dist}")


    def tensor_to_pil(self, tensor: torch.tensor) -> transforms.ToPILImage:
        """Convert tensor image (C, H, W) -> PIL Image."""
        tensor = tensor.detach().cpu().clamp(0, 1)  # Ensure values are in [0, 1]
        return transforms.ToPILImage()(tensor)

    def get_features(self, dataloader: DataLoader, model: torch.nn.Module, transform: transforms.Compose) -> torch.tensor:
        """Extract features from images using evaluation model.

        Args:
            dataloader (DataLoader): DataLoader containing images.
            model (torch.nn.Module): The model used to extract features from the images.
            transform (callable): A function or transform to apply to the images.

        Returns:
        -------
            torch.tensor: A tensor containing the extracted features.

        """
        features = []
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                pil_images = [self.tensor_to_pil(img) for img in images]  # Convert each tensor to PIL
                transformed_images = torch.stack([transform(img) for img in pil_images]).to(self.device)

                feats = model(transformed_images)
                features.append(feats)
        return torch.cat(features, dim=0)

    def get_generated_features(self, model: torch.nn.Module, transform: transforms.Compose) -> torch.tensor:
        """Generate fake images using the generator, apply transformations, and extract features using the provided model.

        Args:
            model (torch.nn.Module): The model used to extract features from the generated images.
            transform (callable): A function or transform to apply to the generated images.

        Returns:
        -------
            torch.tensor: A tensor containing the extracted features.

        """
        self.generator.eval()
        features = []
        with torch.no_grad():
              for label_i, z_i in zip(self.labels, self.z):
                # generate samples for each pair of label and z
                fake_images, _, _ = self.generator_handler.sample_from_generator(batch_size=self.num_class_samples,
                                                                                label=label_i,
                                                                                z=z_i)

                pil_images = [self.tensor_to_pil(img) for img in fake_images]
                transformed_images = torch.stack([transform(img) for img in pil_images]).to(self.device)

                feats = model(transformed_images)
                features.append(feats)
        return torch.cat(features, dim=0)

    pass


    def save_images(self) -> None:
        """Save generated images to disk."""
        self.generator.eval()
        self.generator.to(self.device)

        n_images = self.configs.metrics["save_images"]["n_images"]
        save_path = self.configs.metrics["save_images"]["save_dir"]

        generated_images = []

        i = 0
        labels = set()
        for label_i, z_i in zip(self.labels, self.z):
            if label_i.item() in labels:
                # Skip iteration if it has already been generated
                continue

            if i < n_images:
                # generate samples for each pair of label and z
                generated_sample = self.generator_handler.sample_from_generator(batch_size=1,
                                                                                label=label_i,
                                                                                z=z_i)
                # Save the generated image
                generated_images.append((generated_sample[0], label_i.item(), i))
                i += 1
                labels.add(label_i.item())

        logger.info(f"Generated {len(generated_images)} images from labels {labels}.")
        os.makedirs(save_path, exist_ok=True)
        for img, label, i in generated_images:
            img = img.squeeze(0)
            img = self.tensor_to_pil(img)

            img.save(f"{save_path}/label_{label}_image_{i}.png")

        pass
