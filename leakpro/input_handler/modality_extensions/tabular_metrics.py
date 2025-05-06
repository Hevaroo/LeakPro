"""TabularMetrics class for computing tabular metrics."""
import gower
import numpy as np
import pandas as pd
import torch
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_column import KSComplement
from sdmetrics.single_table import GMLogLikelihood
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

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
            "quality_metrics": self.quality_metrics,
            "plot_densities": self.get_numerical_density_plots,
            "plot_categorical_densities": self.get_categorical_density_plots,
            "gm_likelihood": self.gm_likelihood,
            "gower_distance": self.gower_dist,
        }
        logger.info(configs)
        self.results = {}
        self.numerical_plots = {}
        self.categorical_plots = {}

        self.private_dataloader = self.handler.get_private_dataloader(self.batch_size)
        self.generated_samples = None
        # Compute desired metrics from configs
        # TODO: Change table_name)
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data=self.private_dataloader.dataset)
        # Get columns that are numerical in metadata
        self.numerical_columns = self.metadata.get_column_names(sdtype="numerical")
        self.categorical_columns = self.metadata.get_column_names(sdtype="categorical")
        self.best_rows = pd.DataFrame()
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
            generated_samples, _, _ = self.generator_handler.sample_from_generator(batch_size=self.num_class_samples+1, # TODO: Move to configs asserts, num_class_samples ge 2
                                                                            label=label_i,
                                                                            z=z_i)

            synthetic_label = generated_samples["pseudo_label"].values
            synthetic_label = torch.tensor(synthetic_label, dtype=torch.int).to(self.device)
            generated_samples = generated_samples.drop(columns=["pseudo_label"])
            output = self.evaluation_model(generated_samples)
            prediction = torch.argmax(output, dim=1)
            correct_predictions.append(prediction == synthetic_label)

        correct_predictions = torch.cat(correct_predictions).float()
        self.accuracy = correct_predictions.mean()
        self.accuracy_std = correct_predictions.std() / torch.sqrt(torch.tensor(len(correct_predictions), dtype=torch.float))
        logger.info(f"Accuracy: {self.accuracy.item()}")

        self.results["accuracy"] = self.accuracy.item()
        self.results["accuracy_std"] = self.accuracy_std.item()


    def quality_metrics(self) -> None:
        """Compute quality metrics for the generated samples.

        We generate random samples. In this function, we do not pass the labels and z to the generator.
        """
        self.evaluation_model.eval()
        self.evaluation_model.to(self.device)
        self.generator.eval()
        self.generator.to(self.device)

        # Match the number of samples in the private dataloader
        desired_num_samples = len(self.private_dataloader.dataset)
        self.generated_samples = self.generator.sample(desired_num_samples)

        # We need to change column "pseudo_label" to identity in generated samples to match synthethic and real data
        self.generated_samples["identity"] = self.generated_samples["pseudo_label"]
        self.generated_samples = self.generated_samples.drop(columns=["pseudo_label"])

        logger.info("Computing quality metrics for generated samples.")
        report = QualityReport()
        report.generate(
            real_data=self.private_dataloader.dataset,
            synthetic_data=self.generated_samples,
            metadata=self.metadata.to_dict(),
            verbose=True
        )
        self.results["quality_report"] = report


    def get_numerical_density_plots(self) -> None:
        """Plot the densities of the numerical columns.

        We run this function after quality_metrics.
        """

        if self.numerical_columns is None:
            logger.warning("No numerical columns found in the metadata.")
            return
        for col in self.numerical_columns:
            if col == "identity":
                continue
            self.numerical_plots[f"{col}_density_plot"] = get_column_plot(
                real_data=self.private_dataloader.dataset,
                synthetic_data=self.generated_samples,
                column_name=col,
                metadata=self.metadata
            )

    def get_categorical_density_plots(self) -> None:
        """Plot the densities of the categorical columns.

        We run this function after quality_metrics.
        """
        # categorical_columns = self.categorical_columns
        categorical_columns = ["identity", "race", "insurance", "gender"]
        for col in categorical_columns:
            self.categorical_plots[f"{col}_bar_plot"] = get_column_plot(
                real_data=self.private_dataloader.dataset,
                synthetic_data=self.generated_samples,
                column_name=col,
                metadata=self.metadata,
                plot_type="bar"
            )

    def gm_likelihood(self) -> None:
        """Compute the likelihood of the generated samples using the Gaussian Mixture Model (GMM).

        We run this function after quality_metrics.
        """
        real_data = self.private_dataloader.dataset[self.numerical_columns].drop(columns=["identity"])
        synthetic_data = self.generated_samples[self.numerical_columns].drop(columns=["identity"])
        self.results["gm_likelihood"] = GMLogLikelihood.compute(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )

    def gower_dist(self) -> None:
        """Compute the Gower distance and find best matches between synthethic and real data."""

        if self.generated_samples is None:
            # Match the number of samples in the private dataloader
            desired_num_samples = len(self.private_dataloader.dataset)
            self.generated_samples = self.generator.sample(desired_num_samples)

        table_synthethic = self.generated_samples
        table_real = self.private_dataloader.dataset

        # For each unique value in 'identity' column
        unique_values = table_real["identity"].unique()

        for value in unique_values:
            logger.info(f"Finding best row for identity: {value}")
            # Filter the table for the current value
            syn_subset = table_synthethic[table_synthethic["pseudo_label"] == value].copy()
            real_subset = table_real[table_real["identity"] == value].copy()
            offset = len(syn_subset)

            # Concat the two tables
            table = pd.concat([syn_subset, real_subset], axis=0)

            # Compute the Gower distance
            gower_distance = gower.gower_matrix(table)

            gower_distance = gower_distance[:offset, offset:]

            min_index = np.unravel_index(np.argmin(gower_distance, axis=None), gower_distance.shape)

            # Get the corresponding synthetic and real rows
            synthetic_row = syn_subset.iloc[min_index[0]]
            real_row = real_subset.iloc[min_index[1]]

            synthetic_row = pd.DataFrame(synthetic_row).T
            real_row = pd.DataFrame(real_row).T

            # Cat synthetic and real rows to best_rows dataframe
            self.best_rows = pd.concat([self.best_rows, synthetic_row, real_row], axis=0)
            self.best_rows = self.best_rows.reset_index(drop=True)




