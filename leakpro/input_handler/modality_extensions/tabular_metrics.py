"""TabularMetrics class for computing tabular metrics."""

#import numpy as np
import torch

from leakpro.attacks.utils.generator_handler import GeneratorHandler
from leakpro.input_handler.minv_handler import MINVHandler
from leakpro.utils.logger import logger


class TabularMetrics:
    """Class for computing Tabular metrics."""

    def __init__(self,
                 handler: MINVHandler,
                 generator_handler: GeneratorHandler,
                 configs: dict,
                 labels: torch.tensor = None,
                 z: torch.tensor = None,) -> None:
        """Initialize the TabularMetrics class."""
        self.handler = handler
        self.generator_handler = generator_handler
        self.generator = self.generator_handler.get_generator()
        self.evaluation_model = self.handler.target_model # TODO: Change to evaluation model from configs
        self.target_model = self.handler.target_model
        self.labels = labels
        self.z = z
        logger.info("Configuring TabularMetrics")
        self._configure_metrics(configs)
        self.test_dict = {
            "accuracy": self.compute_accuracy,
        }
        logger.info(configs)
        self.results = {}
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
            generated_samples, _, _ = self.generator_handler.sample_from_generator(batch_size=self.num_class_samples + 1, # TODO: Move to configs asserts, num_class_samples ge 2
                                                                            label=label_i,
                                                                            z=z_i)
            
            generated_samples = generated_samples.drop(columns=["pseudo_label"])
            output = self.evaluation_model(generated_samples)
            prediction = torch.argmax(output, dim=1)
            correct_predictions.append(prediction == label_i)

        correct_predictions = torch.cat(correct_predictions).float()
        self.accuracy = correct_predictions.mean()
        self.accuracy_std = correct_predictions.std() / torch.sqrt(torch.tensor(len(correct_predictions), dtype=torch.float))
        logger.info(f"Accuracy: {self.accuracy.item()}")

        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()
