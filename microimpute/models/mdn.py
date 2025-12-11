"""Mixture Density Network (MDN) imputation model using PyTorch Tabular."""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.models.imputer import (
    Imputer,
    ImputerResults,
    _ConstantValueModel,
)

# PyTorch Tabular imports
try:
    # Set environment variables to suppress logging BEFORE imports
    # pytorch_tabular uses PT_LOGLEVEL to set its log level
    os.environ["PT_LOGLEVEL"] = "ERROR"

    # Suppress lightning rank_zero logging
    for _logger_name in [
        "lightning_utilities.core.rank_zero",
        "pytorch_lightning",
        "pytorch_lightning.utilities.rank_zero",
    ]:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)

    # Suppress pytorch_tabular warnings
    import warnings

    # Suppress all FutureWarnings from pytorch_tabular
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="pytorch_tabular.*",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers.*",
        module="pytorch_lightning",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*argument is set as true but not supported on MPS.*",
        module="torch.utils.data.dataloader",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*training batches.*smaller than the logging interval.*",
        module="pytorch_lightning.loops.fit_loop",
    )

    # After import, also update the rank_zero_module logger
    from lightning_fabric.utilities.rank_zero import rank_zero_module
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import (
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
    )
    from pytorch_tabular.models import CategoryEmbeddingModelConfig, MDNConfig

    rank_zero_module.log.setLevel(logging.ERROR)

    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    PYTORCH_TABULAR_AVAILABLE = False


def _suppress_pytorch_logging() -> None:
    """Suppress verbose logging from PyTorch-related libraries.

    This only suppresses pytorch_tabular and lightning logging,
    leaving microimpute's own logging intact.
    """
    for logger_name in [
        "pytorch_tabular",
        "pytorch_tabular.tabular_model",
        "pytorch_tabular.config",
        "pytorch_tabular.config.config",
        "pytorch_tabular.tabular_datamodule",
        "pytorch_lightning",
        "pytorch_lightning.utilities.rank_zero",
        "lightning",
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning_fabric",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.handlers = []
        logger.addHandler(logging.NullHandler())


def _generate_data_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """Generate a hash from the training data for cache identification.

    Creates a reproducible hash based on the data shape, column names,
    and a sample of the data values.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        A short hash string identifying the dataset.
    """
    # Include shape, column names, and data statistics for identification
    hash_components = [
        str(X.shape),
        str(sorted(X.columns.tolist())),
        str(y.name),
        str(len(y)),
    ]

    # Add hash of actual data values for uniqueness
    # Use pandas hash_pandas_object for consistent hashing
    try:
        data_hash = pd.util.hash_pandas_object(X).sum()
        y_hash = pd.util.hash_pandas_object(y).sum()
        hash_components.extend([str(data_hash), str(y_hash)])
    except Exception:
        # Fallback to basic stats if hashing fails
        hash_components.extend(
            [
                str(X.values.mean()) if X.size > 0 else "0",
                str(y.mean()) if len(y) > 0 else "0",
            ]
        )

    combined = "_".join(hash_components)
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _generate_cache_key(
    predictors: List[str], target: str, data_hash: str
) -> str:
    """Generate a cache key for model storage.

    Args:
        predictors: List of predictor column names.
        target: Target variable name.
        data_hash: Hash of the training data.

    Returns:
        A string cache key combining predictor hash, target, and data hash.
    """
    # Sort predictors for consistent hashing
    sorted_predictors = sorted(predictors)
    predictors_str = "_".join(sorted_predictors)
    # Create a short hash of the predictors
    predictors_hash = hashlib.md5(predictors_str.encode()).hexdigest()[:8]
    # Sanitize target name for filesystem
    safe_target = target.replace("/", "_").replace("\\", "_")
    return f"{predictors_hash}_{safe_target}_{data_hash}"


class _MDNModel:
    """Internal wrapper for PyTorch Tabular MDN model for numeric targets."""

    def __init__(
        self,
        seed: int,
        logger,
        layers: str = "128-64-32",
        activation: str = "ReLU",
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        num_gaussian: int = 5,
        softmax_temperature: float = 1.0,
        n_samples: int = 100,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        batch_size: int = 256,
    ):
        self.seed = seed
        self.logger = logger
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_gaussian = num_gaussian
        self.softmax_temperature = softmax_temperature
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size

        self.model = None
        self.output_column = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_cols: Optional[List[str]] = None,
    ) -> None:
        """Fit the MDN model.

        Args:
            X: Feature DataFrame (predictors are already dummy-encoded).
            y: Target Series.
            categorical_cols: Optional list of column names that are dummy-encoded
                categorical variables. These will be passed to PyTorch Tabular
                as categorical columns for embedding.
        """
        _suppress_pytorch_logging()
        self.output_column = y.name
        categorical_cols = categorical_cols or []

        # Combine X and y for PyTorch Tabular
        train_data = X.copy()
        train_data[y.name] = y

        # Separate continuous and categorical columns
        # Categorical columns get embeddings, continuous columns are passed as-is
        continuous_cols = [
            col for col in X.columns.tolist() if col not in categorical_cols
        ]

        # Configure data
        data_config = DataConfig(
            target=[y.name],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )

        # Configure trainer
        # Note: checkpoints and load_best disabled to avoid PyTorch 2.6
        # weights_only=True issue with OmegaConf objects
        trainer_config = TrainerConfig(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping="valid_loss",
            early_stopping_patience=self.early_stopping_patience,
            checkpoints=None,
            load_best=False,
            accelerator="auto",
            devices=1,
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )

        # Configure optimizer
        optimizer_config = OptimizerConfig(
            optimizer="Adam",
            lr_scheduler=None,
        )

        # Configure MDN model
        model_config = MDNConfig(
            task="regression",
            backbone_config_class="CategoryEmbeddingModelConfig",
            backbone_config_params={
                "task": "backbone",  # Required: tells it to act as feature extractor
                "layers": self.layers,
                "activation": self.activation,
                "dropout": self.dropout,
                "use_batch_norm": self.use_batch_norm,
            },
            head_config={
                "num_gaussian": self.num_gaussian,
                "softmax_temperature": self.softmax_temperature,
                "n_samples": self.n_samples,
            },
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

        # Create and train model
        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )

        self.model.fit(train=train_data)

    def predict(self, X: pd.DataFrame, n_samples: int = 1) -> np.ndarray:
        """Sample from the MDN distribution.

        Predictions are made by stochastically sampling from the learned
        mixture distribution, not by returning point estimates.

        Args:
            X: Feature DataFrame.
            n_samples: Number of samples per observation.

        Returns:
            Array of shape (n_observations,) if n_samples=1, or
            (n_observations, n_samples) with sampled values.
        """
        # Put model in eval mode
        self.model.model.eval()

        # Create inference dataloader
        test_loader = self.model.datamodule.prepare_inference_dataloader(X)

        samples_list = []
        with torch.no_grad():
            for batch in test_loader:
                # Move batch tensors to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.model.model.device)

                # Sample from the mixture distribution
                samples = self.model.model.sample(batch, n_samples=n_samples)

                if n_samples == 1:
                    samples_list.append(samples.squeeze(-1).cpu().numpy())
                else:
                    samples_list.append(samples.cpu().numpy())

        return np.concatenate(samples_list)

    def save(self, path: str) -> None:
        """Save the model to disk."""
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str, seed: int, logger) -> "_MDNModel":
        """Load a model from disk."""
        instance = cls(seed=seed, logger=logger)
        instance.model = TabularModel.load_model(path)
        return instance


class _NeuralClassifierModel:
    """Internal wrapper for PyTorch Tabular classifier for categorical/boolean."""

    def __init__(
        self,
        seed: int,
        logger,
        layers: str = "128-64-32",
        activation: str = "ReLU",
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        batch_size: int = 256,
    ):
        self.seed = seed
        self.logger = logger
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size

        self.model = None
        self.output_column = None
        self.var_type = None
        self.categories = None
        self.label_map = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        var_type: str,
        categories: Optional[List] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> None:
        """Fit the neural classifier.

        Args:
            X: Feature DataFrame (predictors are already dummy-encoded).
            y: Target Series (original categorical/boolean column).
            var_type: Type of variable ("boolean" or "categorical").
            categories: List of categories for categorical variables.
            categorical_cols: Optional list of column names that are dummy-encoded
                categorical variables. These will be passed to PyTorch Tabular
                as categorical columns for embedding.
        """
        _suppress_pytorch_logging()
        self.output_column = y.name
        self.var_type = var_type
        categorical_cols = categorical_cols or []

        if var_type == "boolean":
            y_encoded = y.astype(int)
            self.categories = [False, True]
        else:
            self.categories = categories if categories else y.unique().tolist()
            self.label_map = {cat: i for i, cat in enumerate(self.categories)}
            y_encoded = y.map(self.label_map)

            if y_encoded.isna().any():
                self.logger.warning(
                    f"Found {y_encoded.isna().sum()} unmapped values in "
                    f"{self.output_column}"
                )
                y_encoded = y_encoded.fillna(0)

        # Combine X and encoded y for PyTorch Tabular
        train_data = X.copy()
        train_data[y.name] = y_encoded.astype(int)

        # Separate continuous and categorical columns
        # Categorical columns get embeddings, continuous columns are passed as-is
        continuous_cols = [
            col for col in X.columns.tolist() if col not in categorical_cols
        ]

        # Configure data
        data_config = DataConfig(
            target=[y.name],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )

        # Configure trainer
        # Note: checkpoints and load_best disabled to avoid PyTorch 2.6
        # weights_only=True issue with OmegaConf objects
        trainer_config = TrainerConfig(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            early_stopping="valid_loss",
            early_stopping_patience=self.early_stopping_patience,
            checkpoints=None,
            load_best=False,
            accelerator="auto",
            devices=1,
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )

        # Configure optimizer
        optimizer_config = OptimizerConfig(
            optimizer="Adam",
            lr_scheduler=None,
        )

        # Configure classifier model with LinearHead
        model_config = CategoryEmbeddingModelConfig(
            task="classification",
            layers=self.layers,
            activation=self.activation,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

        # Create and train model
        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )

        self.model.fit(train=train_data)

    def predict(
        self,
        X: pd.DataFrame,
        return_probs: bool = False,
    ) -> Union[pd.Series, Dict[str, Any]]:
        """Predict classes by stochastically sampling from probability distribution.

        Predictions are made by sampling from the predicted probability
        distribution, not by returning the argmax class.

        Args:
            X: Input features.
            return_probs: If True, return probability distributions.

        Returns:
            Predicted values as Series, or dict with probabilities if
            return_probs=True.
        """
        # Get predictions with probabilities
        preds_df = self.model.predict(X, ret_logits=False)

        # Extract probability columns (named like target_0_probability, etc.)
        prob_cols = sorted(
            [c for c in preds_df.columns if c.endswith("_probability")]
        )
        probs = preds_df[prob_cols].values

        if return_probs:
            if self.var_type == "boolean":
                original_classes = [False, True]
            else:
                original_classes = self.categories

            return {
                "probabilities": probs,
                "classes": np.array(original_classes),
            }

        # Stochastically sample from probability distribution
        rng = np.random.default_rng(self.seed)
        sampled_indices = np.array(
            [rng.choice(len(self.categories), p=p) for p in probs]
        )

        if self.var_type == "boolean":
            predictions = pd.Series(
                sampled_indices.astype(bool), index=X.index
            )
        else:
            predictions = pd.Series(
                [self.categories[i] for i in sampled_indices], index=X.index
            )

        predictions.name = self.output_column
        return predictions

    def save(self, path: str) -> None:
        """Save the model to disk."""
        self.model.save_model(path)
        # Also save metadata
        import json

        metadata = {
            "output_column": self.output_column,
            "var_type": self.var_type,
            "categories": [
                str(c) if not isinstance(c, (bool, int, float)) else c
                for c in self.categories
            ],
            "label_map": self.label_map,
        }
        with open(os.path.join(path, "classifier_metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str, seed: int, logger) -> "_NeuralClassifierModel":
        """Load a model from disk."""
        import json

        instance = cls(seed=seed, logger=logger)
        instance.model = TabularModel.load_model(path)

        # Load metadata
        with open(os.path.join(path, "classifier_metadata.json"), "r") as f:
            metadata = json.load(f)

        instance.output_column = metadata["output_column"]
        instance.var_type = metadata["var_type"]
        instance.categories = metadata["categories"]
        instance.label_map = metadata["label_map"]

        # Convert boolean strings back
        if instance.var_type == "boolean":
            instance.categories = [False, True]

        return instance


class MDNResults(ImputerResults):
    """Fitted MDN model container ready for imputation."""

    def __init__(
        self,
        models: Dict[str, Any],
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, str]] = None,
        original_predictors: Optional[List[str]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        dummy_processor: Optional[Any] = None,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize MDN results.

        Args:
            models: Dictionary of fitted models for each variable.
            predictors: List of predictor variable names.
            imputed_variables: List of imputed variable names.
            seed: Random seed for reproducibility.
            imputed_vars_dummy_info: Optional dummy variable info.
            original_predictors: Original predictor names before encoding.
            categorical_targets: Dictionary of categorical target info.
            boolean_targets: Dictionary of boolean target info.
            constant_targets: Dictionary of constant target info.
            dummy_processor: Processor for handling dummy encoding.
            log_level: Logging level.
        """
        super().__init__(
            predictors,
            imputed_variables,
            seed,
            imputed_vars_dummy_info,
            original_predictors,
            log_level,
        )
        self.models = models
        self.categorical_targets = categorical_targets or {}
        self.boolean_targets = boolean_targets or {}
        self.constant_targets = constant_targets or {}
        self.dummy_processor = dummy_processor

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
        n_samples: int = 1000,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values using stochastic sampling.

        For MDN models, many samples are drawn from the learned mixture
        distribution and empirical quantiles are computed. For classifier
        models, samples are drawn from the predicted probability distribution.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to compute from the sampled
                distribution.
            return_probs: If True, return probability distributions for
                categorical variables.
            n_samples: Number of samples to draw for computing quantiles
                (default 1000). More samples give more accurate quantile
                estimates but increase computation time.

        Returns:
            Dictionary mapping quantiles to imputed DataFrames.
            If return_probs=True, includes 'probabilities' key.
        """
        try:
            imputations: Dict[float, pd.DataFrame] = {}
            prob_results = {} if return_probs else None

            if quantiles:
                quantiles_to_use = quantiles
            else:
                quantiles_to_use = [0.5]

            # Pre-compute samples for MDN models (draw once, compute all
            # quantiles)
            mdn_samples: Dict[str, np.ndarray] = {}
            for variable in self.imputed_variables:
                model = self.models[variable]
                if isinstance(model, _MDNModel):
                    # Draw n_samples for each observation
                    samples = model.predict(
                        X_test[self.predictors], n_samples=n_samples
                    )
                    mdn_samples[variable] = samples

            # Compute quantiles from the samples
            for q in quantiles_to_use:
                imputed_df = pd.DataFrame(index=X_test.index)

                for variable in self.imputed_variables:
                    model = self.models[variable]

                    if isinstance(model, _ConstantValueModel):
                        imputed_df[variable] = model.predict(X_test)

                    elif isinstance(model, _NeuralClassifierModel):
                        # Stochastic sampling from probability distribution
                        if return_probs and prob_results is not None:
                            prob_info = model.predict(
                                X_test[self.predictors],
                                return_probs=True,
                            )
                            prob_results[variable] = prob_info

                        imputed_df[variable] = model.predict(
                            X_test[self.predictors],
                            return_probs=False,
                        )

                    elif isinstance(model, _MDNModel):
                        # Compute empirical quantile from samples
                        samples = mdn_samples[variable]
                        quantile_values = np.quantile(samples, q, axis=1)
                        imputed_df[variable] = quantile_values

                    else:
                        raise ValueError(
                            f"Unknown model type for variable {variable}"
                        )

                imputations[q] = imputed_df

            # Add probabilities if requested
            if return_probs and prob_results:
                imputations["probabilities"] = prob_results

            # Return format based on whether quantiles were specified
            if quantiles is not None:
                return imputations
            else:
                return imputations[0.5]

        except Exception as e:
            self.logger.error(f"Error during MDN prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with MDN model: {str(e)}"
            ) from e


class MDN(Imputer):
    """
    Mixture Density Network imputer using PyTorch Tabular.

    This imputer uses a neural network with a Mixture Density Network head
    for numeric targets and a neural classifier with LinearHead for
    categorical/boolean targets. Both share the same backbone architecture.

    Predictions are made by stochastically sampling from the learned
    distributions rather than returning point estimates.

    Models are automatically cached based on a hash of the training data,
    so identical datasets will reuse previously trained models.

    Attributes:
        layers: Network architecture as hyphen-separated string (e.g., "128-64").
        activation: Activation function name (e.g., "ReLU", "LeakyReLU").
        dropout: Dropout probability.
        use_batch_norm: Whether to use batch normalization.
        num_gaussian: Number of Gaussian components in MDN mixture.
        softmax_temperature: Temperature for mixture weight softmax.
        n_samples: Number of samples for MDN prediction.
        learning_rate: Learning rate for training.
        max_epochs: Maximum training epochs.
        early_stopping_patience: Patience for early stopping.
        batch_size: Training batch size.
        model_dir: Directory for saving/loading models.
        force_retrain: If True, always retrain instead of loading cached models.
    """

    def __init__(
        self,
        # Backbone config (shared)
        layers: str = "128-64-32",
        activation: str = "ReLU",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        # MDN Head config
        num_gaussian: int = 5,
        softmax_temperature: float = 1.0,
        n_samples: int = 100,
        # Training config
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        batch_size: int = 256,
        # Caching config
        model_dir: str = "./microimpute_models",
        force_retrain: bool = False,
        # Standard imputer params
        seed: Optional[int] = RANDOM_STATE,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the MDN imputer.

        Args:
            layers: Network architecture (e.g., "128-64-32" for three layers).
            activation: Activation function (ReLU, LeakyReLU, etc.).
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
            num_gaussian: Number of Gaussian components in mixture.
            softmax_temperature: Temperature for mixture weights.
            n_samples: Number of samples for prediction.
            learning_rate: Learning rate for Adam optimizer.
            max_epochs: Maximum training epochs.
            early_stopping_patience: Early stopping patience.
            batch_size: Training batch size.
            model_dir: Directory for saving/loading models.
            force_retrain: If True, skip cache and always retrain.
            seed: Random seed for reproducibility.
            log_level: Logging level.
        """
        if not PYTORCH_TABULAR_AVAILABLE:
            raise ImportError(
                "pytorch-tabular is required for MDN imputer. "
                "Install it with: pip install pytorch_tabular"
            )

        super().__init__(seed=seed, log_level=log_level)

        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.num_gaussian = num_gaussian
        self.softmax_temperature = softmax_temperature
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.force_retrain = force_retrain
        self.log_level = log_level

        self.models = {}

        self.logger.debug("Initializing MDN imputer")

    def _get_cache_path(
        self, predictors: List[str], target: str, data_hash: str
    ) -> str:
        """Get the cache path for a model.

        Args:
            predictors: List of predictor column names.
            target: Target variable name.
            data_hash: Hash of the training data.

        Returns:
            Path to the model cache directory.
        """
        cache_key = _generate_cache_key(predictors, target, data_hash)
        return os.path.join(self.model_dir, cache_key)

    def _model_exists(self, cache_path: Optional[str]) -> bool:
        """Check if a cached model exists.

        Args:
            cache_path: Path to check.

        Returns:
            True if model exists at path.
        """
        if cache_path is None:
            return False
        return os.path.exists(cache_path) and os.path.isdir(cache_path)

    @validate_call(config=VALIDATE_CONFIG)
    def _fit(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        original_predictors: Optional[List[str]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        numeric_targets: Optional[List[str]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        tune_hyperparameters: bool = False,
        **kwargs: Any,
    ) -> Union[MDNResults, Tuple[MDNResults, Dict[str, Any]]]:
        """Fit the MDN model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            original_predictors: Original predictor names before encoding.
            categorical_targets: Dict of categorical target info.
            boolean_targets: Dict of boolean target info.
            numeric_targets: List of numeric target names.
            constant_targets: Dict of constant target info.
            tune_hyperparameters: If True, tune hyperparameters before fitting.
            **kwargs: Additional parameters.

        Returns:
            MDNResults instance with fitted models.
            If tune_hyperparameters=True, returns (MDNResults, best_params).
        """
        try:
            best_params = None

            # Optionally tune hyperparameters before fitting
            if tune_hyperparameters:
                self.logger.info("Starting hyperparameter tuning...")
                best_params = self._tune_hyperparameters(
                    X_train,
                    predictors,
                    imputed_variables,
                    categorical_targets,
                    boolean_targets,
                )

                # Apply tuned parameters
                if isinstance(best_params, dict):
                    if "mdn" in best_params:
                        # Mixed types - apply MDN params
                        mdn_params = best_params["mdn"]
                        if "num_gaussian" in mdn_params:
                            self.num_gaussian = mdn_params["num_gaussian"]
                        if "learning_rate" in mdn_params:
                            self.learning_rate = mdn_params["learning_rate"]
                        # Classifier params stored separately
                    elif "classifier" in best_params:
                        # Only categorical - apply to learning rate
                        classifier_params = best_params["classifier"]
                        if "learning_rate" in classifier_params:
                            self.learning_rate = classifier_params[
                                "learning_rate"
                            ]
                    else:
                        # Only numeric - flat dict
                        if "num_gaussian" in best_params:
                            self.num_gaussian = best_params["num_gaussian"]
                        if "learning_rate" in best_params:
                            self.learning_rate = best_params["learning_rate"]

                self.logger.info(
                    f"Applied tuned hyperparameters: {best_params}"
                )

            self.logger.info(
                f"Fitting MDN model with {len(predictors)} predictors"
            )

            self.models = {}

            # If force_retrain is True, delete existing cached models
            if self.force_retrain and Path(self.model_dir).exists():
                shutil.rmtree(self.model_dir)
                self.logger.info(
                    f"Deleted cached models directory: {self.model_dir}"
                )

            # Ensure model directory exists
            Path(self.model_dir).mkdir(parents=True, exist_ok=True)

            # Extract categorical columns from dummy_processor
            categorical_cols = []
            if hasattr(self, "dummy_processor") and self.dummy_processor:
                for dummy_cols in self.dummy_processor.dummy_mapping.values():
                    categorical_cols.extend(dummy_cols)

            for variable in imputed_variables:
                # Handle constant targets (no caching needed)
                if variable in (constant_targets or {}):
                    constant_val = constant_targets[variable]["value"]
                    model = _ConstantValueModel(constant_val, variable)
                    self.models[variable] = model
                    self.logger.info(
                        f"Using constant value {constant_val} for {variable}"
                    )
                    continue

                # Generate data hash for caching
                Y = X_train[variable]
                data_hash = _generate_data_hash(X_train[predictors], Y)
                cache_path = self._get_cache_path(
                    predictors, variable, data_hash
                )

                # Check cache
                if not self.force_retrain and self._model_exists(cache_path):
                    self.logger.info(
                        f"Loading cached model for '{variable}' from {cache_path}"
                    )
                    try:
                        if variable in (
                            categorical_targets or {}
                        ) or variable in (boolean_targets or {}):
                            model = _NeuralClassifierModel.load(
                                cache_path, self.seed, self.logger
                            )
                        else:
                            model = _MDNModel.load(
                                cache_path, self.seed, self.logger
                            )
                        self.models[variable] = model
                        continue
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load cached model: {e}. Retraining..."
                        )

                # Choose model based on variable type
                if variable in (categorical_targets or {}):
                    self.logger.warning(
                        f"Using neural network for categorical variable "
                        f"'{variable}'. This may be computationally expensive "
                        f"for simple classification tasks."
                    )

                    model = _NeuralClassifierModel(
                        seed=self.seed,
                        logger=self.logger,
                        layers=self.layers,
                        activation=self.activation,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                        learning_rate=self.learning_rate,
                        max_epochs=self.max_epochs,
                        early_stopping_patience=self.early_stopping_patience,
                        batch_size=self.batch_size,
                    )
                    model.fit(
                        X_train[predictors],
                        Y,
                        var_type=categorical_targets[variable]["type"],
                        categories=categorical_targets[variable].get(
                            "categories"
                        ),
                        categorical_cols=categorical_cols,
                    )
                    self.logger.info(
                        f"Neural classifier fitted for categorical variable "
                        f"{variable}"
                    )

                elif variable in (boolean_targets or {}):
                    self.logger.warning(
                        f"Using neural network for boolean variable "
                        f"'{variable}'. This may be computationally expensive "
                        f"for simple classification tasks."
                    )

                    model = _NeuralClassifierModel(
                        seed=self.seed,
                        logger=self.logger,
                        layers=self.layers,
                        activation=self.activation,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                        learning_rate=self.learning_rate,
                        max_epochs=self.max_epochs,
                        batch_size=self.batch_size,
                    )
                    model.fit(
                        X_train[predictors],
                        Y,
                        var_type="boolean",
                        categorical_cols=categorical_cols,
                    )
                    self.logger.info(
                        f"Neural classifier fitted for boolean variable "
                        f"{variable}"
                    )

                else:
                    # Numeric target - use MDN
                    self.logger.info(
                        f"Training MDN for numeric variable {variable}"
                    )

                    model = _MDNModel(
                        seed=self.seed,
                        logger=self.logger,
                        layers=self.layers,
                        activation=self.activation,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                        num_gaussian=self.num_gaussian,
                        softmax_temperature=self.softmax_temperature,
                        n_samples=self.n_samples,
                        learning_rate=self.learning_rate,
                        max_epochs=self.max_epochs,
                        batch_size=self.batch_size,
                    )
                    model.fit(
                        X_train[predictors],
                        Y,
                        categorical_cols=categorical_cols,
                    )
                    self.logger.info(
                        f"MDN fitted for numeric variable {variable}"
                    )

                self.models[variable] = model

                # Save to cache
                try:
                    Path(cache_path).mkdir(parents=True, exist_ok=True)
                    model.save(cache_path)
                    self.logger.info(f"Saved model to {cache_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save model: {e}")

            results = MDNResults(
                models=self.models,
                predictors=predictors,
                imputed_variables=imputed_variables,
                imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                original_predictors=self.original_predictors,
                categorical_targets=categorical_targets,
                boolean_targets=boolean_targets,
                constant_targets=constant_targets,
                dummy_processor=getattr(self, "dummy_processor", None),
                seed=self.seed,
                log_level=self.log_level,
            )

            # Return tuple if hyperparameter tuning was performed
            if tune_hyperparameters and best_params is not None:
                return results, best_params
            return results

        except Exception as e:
            self.logger.error(f"Error fitting MDN model: {str(e)}")
            raise RuntimeError(f"Failed to fit MDN model: {str(e)}") from e

    def _tune_mdn_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        numeric_vars: List[str],
        n_cv_folds: int = 3,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Tune MDN hyperparameters using Optuna with cross-validation.

        Tunes num_gaussian and learning_rate for numeric targets.

        Args:
            data: Full training data.
            predictors: List of column names to use as predictors.
            numeric_vars: List of numeric variables to impute.
            n_cv_folds: Number of CV folds (default: 3).
            n_trials: Number of Optuna trials (default: 10).

        Returns:
            Dictionary of tuned hyperparameters for MDN.
        """
        import optuna
        from sklearn.model_selection import KFold

        from microimpute.comparisons.metrics import compute_loss

        # Suppress Optuna's logs during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Set up CV folds
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.seed)

        def objective(trial: optuna.Trial) -> float:
            # Only tune num_gaussian and learning_rate
            num_gaussian = trial.suggest_int("num_gaussian", 2, 10)
            learning_rate = trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            )

            # Track errors across CV folds
            fold_errors = []

            # Perform CV
            for train_idx, val_idx in kf.split(data):
                X_train_fold = data.iloc[train_idx]
                X_val_fold = data.iloc[val_idx]

                # Track errors for numeric variables in this fold
                var_errors = []

                for var in numeric_vars:
                    y_train = X_train_fold[var]
                    y_val = X_val_fold[var]

                    # Create and fit MDN model with trial parameters
                    model = _MDNModel(
                        seed=self.seed,
                        logger=self.logger,
                        layers=self.layers,
                        activation=self.activation,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                        num_gaussian=num_gaussian,
                        softmax_temperature=self.softmax_temperature,
                        n_samples=self.n_samples,
                        learning_rate=learning_rate,
                        max_epochs=40,  # Reduced for tuning
                        batch_size=self.batch_size,
                    )
                    model.fit(X_train_fold[predictors], y_train)

                    # Predict using stochastic sampling
                    y_pred = model.predict(X_val_fold[predictors], n_samples=1)

                    # Use quantile loss with median (q=0.5) for tuning
                    _, quantile_loss_value = compute_loss(
                        y_val.values.flatten(),
                        y_pred.flatten(),
                        "quantile_loss",
                        q=0.5,
                    )

                    # Normalize by variable's standard deviation
                    std = np.std(y_val.values.flatten())
                    normalized_loss = (
                        quantile_loss_value / std
                        if std > 0
                        else quantile_loss_value
                    )

                    var_errors.append(normalized_loss)

                # Average across variables for this fold
                if var_errors:
                    fold_errors.append(np.mean(var_errors))

            # Return mean error across all CV folds
            return np.mean(fold_errors) if fold_errors else float("inf")

        # Create and run the study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        study.optimize(objective, n_trials=n_trials)

        best_value = study.best_value
        self.logger.info(
            f"MDN - Lowest average normalized quantile loss "
            f"({n_cv_folds}-fold CV): {best_value}"
        )

        best_params = study.best_params
        self.logger.info(f"MDN - Best hyperparameters found: {best_params}")

        return best_params

    def _tune_classifier_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        categorical_vars: List[str],
        categorical_targets: Optional[Dict] = None,
        boolean_targets: Optional[Dict] = None,
        n_cv_folds: int = 3,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Tune neural classifier hyperparameters using Optuna with CV.

        Tunes learning_rate for categorical/boolean targets.

        Args:
            data: Full training data.
            predictors: List of column names to use as predictors.
            categorical_vars: List of categorical/boolean variables to impute.
            categorical_targets: Dict of categorical target info.
            boolean_targets: Dict of boolean target info.
            n_cv_folds: Number of CV folds (default: 3).
            n_trials: Number of Optuna trials (default: 10).

        Returns:
            Dictionary of tuned hyperparameters for classifier.
        """
        import optuna
        from sklearn.model_selection import KFold

        from microimpute.comparisons.metrics import (
            compute_loss,
            order_probabilities_alphabetically,
        )

        # Suppress Optuna's logs during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        categorical_targets = categorical_targets or {}
        boolean_targets = boolean_targets or {}

        # Set up CV folds
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.seed)

        def objective(trial: optuna.Trial) -> float:
            # Only tune learning_rate for classifier
            learning_rate = trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            )

            # Track errors across CV folds
            fold_errors = []

            # Perform CV
            for train_idx, val_idx in kf.split(data):
                X_train_fold = data.iloc[train_idx]
                X_val_fold = data.iloc[val_idx]

                # Track errors for categorical variables in this fold
                var_errors = []

                for var in categorical_vars:
                    y_train = X_train_fold[var]
                    y_val = X_val_fold[var]

                    # Determine variable type
                    if var in boolean_targets:
                        var_type = "boolean"
                        categories = None
                    else:
                        var_type = categorical_targets[var].get(
                            "type", "categorical"
                        )
                        categories = categorical_targets[var].get("categories")

                    # Create and fit classifier with trial parameters
                    # Use 40 epochs for tuning
                    model = _NeuralClassifierModel(
                        seed=self.seed,
                        logger=self.logger,
                        layers=self.layers,
                        activation=self.activation,
                        dropout=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                        learning_rate=learning_rate,
                        max_epochs=40,  # Reduced for tuning
                        batch_size=self.batch_size,
                    )
                    model.fit(
                        X_train_fold[predictors],
                        y_train,
                        var_type=var_type,
                        categories=categories,
                    )

                    # Get probabilities for log loss calculation
                    prob_info = model.predict(
                        X_val_fold[predictors], return_probs=True
                    )
                    probs = prob_info["probabilities"]
                    classes = prob_info["classes"]

                    # Order probabilities alphabetically for consistent evaluation
                    probs, classes = order_probabilities_alphabetically(
                        probs, classes
                    )

                    # Compute log loss
                    _, log_loss_value = compute_loss(
                        y_val.values,
                        probs,
                        "log_loss",
                        labels=classes,
                    )

                    var_errors.append(log_loss_value)

                # Average across variables for this fold
                if var_errors:
                    fold_errors.append(np.mean(var_errors))

            # Return mean error across all CV folds
            return np.mean(fold_errors) if fold_errors else float("inf")

        # Create and run the study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        study.optimize(objective, n_trials=n_trials)

        best_value = study.best_value
        self.logger.info(
            f"Classifier - Lowest average log loss "
            f"({n_cv_folds}-fold CV): {best_value}"
        )

        best_params = study.best_params
        self.logger.info(
            f"Classifier - Best hyperparameters found: {best_params}"
        )

        return best_params

    def _tune_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        categorical_targets: Optional[Dict] = None,
        boolean_targets: Optional[Dict] = None,
        n_cv_folds: int = 3,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Coordinate hyperparameter tuning for MDN.

        Automatically detects variable types and tunes appropriate models:
        - Numeric variables: MDN with num_gaussian and learning_rate
        - Categorical/Boolean variables: Classifier with learning_rate

        Args:
            data: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            categorical_targets: Dict of categorical target info.
            boolean_targets: Dict of boolean target info.

        Returns:
            Dictionary of tuned hyperparameters. Format depends on variable types:
            - Only numeric: flat dict with MDN params
            - Only categorical: flat dict with classifier params
            - Mixed: nested dict {"mdn": {...}, "classifier": {...}}
        """
        categorical_targets = categorical_targets or {}
        boolean_targets = boolean_targets or {}

        # Separate variables by type
        categorical_vars = [
            var
            for var in imputed_variables
            if var in categorical_targets or var in boolean_targets
        ]
        numeric_vars = [
            var for var in imputed_variables if var not in categorical_vars
        ]

        self.logger.info(
            f"MDN hyperparameter tuning with {n_cv_folds}-fold CV and "
            f"{n_trials} trials: {len(numeric_vars)} numeric variables, "
            f"{len(categorical_vars)} categorical/boolean variables"
        )

        # Tune appropriate models based on variable types
        if not categorical_vars:
            # Only numeric variables
            self.logger.info(
                "Tuning MDN hyperparameters (numeric variables only)"
            )
            return self._tune_mdn_hyperparameters(
                data, predictors, numeric_vars, n_cv_folds, n_trials
            )
        elif not numeric_vars:
            # Only categorical variables
            self.logger.info(
                "Tuning classifier hyperparameters "
                "(categorical/boolean variables only)"
            )
            return self._tune_classifier_hyperparameters(
                data,
                predictors,
                categorical_vars,
                categorical_targets,
                boolean_targets,
                n_cv_folds,
                n_trials,
            )
        else:
            # Mixed: tune both separately
            self.logger.info(
                "Tuning both MDN and classifier hyperparameters "
                "(mixed variable types)"
            )
            mdn_params = self._tune_mdn_hyperparameters(
                data, predictors, numeric_vars, n_cv_folds, n_trials
            )
            classifier_params = self._tune_classifier_hyperparameters(
                data,
                predictors,
                categorical_vars,
                categorical_targets,
                boolean_targets,
                n_cv_folds,
                n_trials,
            )
            return {"mdn": mdn_params, "classifier": classifier_params}
