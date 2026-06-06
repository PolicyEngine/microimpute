"""Quantile Regression Forest imputation model with sequential imputation."""

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestClassifier

from microimpute.config import VALIDATE_CONFIG
from microimpute.models.imputer import BaseImputer, ImputerResults

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _get_sequential_predictors(
    predictors: List[str],
    imputed_variables: List[str],
    current_variable_index: int,
) -> List[str]:
    """Get the predictor set for sequential imputation.

    Args:
        predictors: Original predictor variables
        imputed_variables: Variables being imputed
        current_variable_index: Index of the current variable being imputed

    Returns:
        List of predictor columns including previously imputed variables
    """
    return predictors + imputed_variables[:current_variable_index]


class _RandomForestClassifierModel:
    """Internal class to handle classification for categorical/boolean targets."""

    def __init__(self, seed: int, logger):
        self.seed = seed
        self.logger = logger
        self.classifier = None
        self.output_column = None
        self.var_type = None
        self.categories = None
        self.label_map = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        var_type: str,
        categories: List = None,
        sample_weight: Optional[np.ndarray] = None,
        **rf_kwargs: Any,
    ) -> None:
        """Fit classifier for categorical/boolean target.

        Note: y should be the ORIGINAL categorical/boolean column,
        not dummy encoded.
        """
        self.output_column = y.name
        self.var_type = var_type

        if var_type == "boolean":
            # For boolean, convert to 0/1 but keep as single target
            y_encoded = y.astype(int)
            self.categories = [False, True]
        else:
            # For categorical, create label encoding
            self.categories = categories if categories else y.unique().tolist()
            self.label_map = {cat: i for i, cat in enumerate(self.categories)}
            y_encoded = y.map(self.label_map)

            # Check for unmapped values
            if y_encoded.isna().any():
                self.logger.warning(
                    f"Found {y_encoded.isna().sum()} unmapped values in {self.output_column}"
                )
                y_encoded = y_encoded.fillna(0)  # Default to first category

        # Extract relevant RF parameters from kwargs
        classifier_params = {
            "n_estimators": rf_kwargs.get("n_estimators", 100),
            "max_depth": rf_kwargs.get("max_depth", None),
            "min_samples_split": rf_kwargs.get("min_samples_split", 2),
            "min_samples_leaf": rf_kwargs.get("min_samples_leaf", 1),
            "max_features": rf_kwargs.get("max_features", "sqrt"),
            "random_state": self.seed,
        }

        self.classifier = RandomForestClassifier(**classifier_params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(
                sample_weight, dtype=float
            )
        self.classifier.fit(X, y_encoded, **fit_kwargs)

    def predict(
        self, X: pd.DataFrame, return_probs: bool = False
    ) -> pd.Series:
        """Predict classes or probabilities."""
        if return_probs:
            probs = self.classifier.predict_proba(X)
            # Return both probabilities and the original category labels
            # The probabilities are ordered according to self.classifier.classes_
            # which are the encoded values, but we need to return the original labels
            # in the same order

            if self.var_type == "boolean":
                # For boolean, classes are simply False and True
                # sklearn's classifier.classes_ will be [0, 1] in order
                original_classes = [False, True]
            else:
                # For categorical, map encoded values back to original labels
                original_classes = []
                for encoded_val in self.classifier.classes_:
                    # Find the original category for this encoded value
                    for cat, enc in self.label_map.items():
                        if enc == encoded_val:
                            original_classes.append(cat)
                            break

            return {
                "probabilities": probs,
                "classes": np.array(original_classes),
            }
        else:
            y_pred = self.classifier.predict(X)

            if self.var_type == "boolean":
                predictions = pd.Series(y_pred.astype(bool), index=X.index)
            else:
                # Map back to original categories
                reverse_map = {i: cat for cat, i in self.label_map.items()}
                predictions = pd.Series(y_pred).map(reverse_map)
                predictions.index = X.index

            predictions.name = self.output_column
            return predictions


class _QRFModel:
    """Internal class to handle QRF model with quantile prediction logic."""

    def __init__(self, seed: int, logger):
        self.seed = seed
        self.logger = logger
        self.qrf = None
        self.output_column = None
        # Create the RNG once at construction so that repeated predict()
        # calls consume state progressively and return different draws.
        self._rng = np.random.default_rng(self.seed)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        **qrf_kwargs: Any,
    ) -> None:
        """Fit the QRF model.

        Note: Assumes X is already preprocessed with categorical encoding
        handled by the base imputer class.

        Args:
            X: Predictor DataFrame (preprocessed).
            y: Target Series.
            sample_weight: Optional per-row sample weights, passed directly to
                the underlying ``RandomForestQuantileRegressor.fit`` so each
                row contributes to the weighted-survey estimator rather than
                being treated as a bootstrap-resample probability.
        """
        self.output_column = y.name

        # Remove random_state / sample_weight from kwargs if present, since
        # we set them explicitly below.
        qrf_kwargs_filtered = {
            k: v
            for k, v in qrf_kwargs.items()
            if k not in ("random_state", "sample_weight")
        }

        # Create and fit model
        self.qrf = RandomForestQuantileRegressor(
            random_state=self.seed, **qrf_kwargs_filtered
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(
                sample_weight, dtype=float
            )
        self.qrf.fit(X, y.values.ravel(), **fit_kwargs)

    def predict(
        self,
        X: pd.DataFrame,
        mean_quantile: float = 0.5,
        count_samples: int = 10,
        exact_quantile: Optional[float] = None,
    ) -> pd.Series:
        """Predict using the fitted model with beta distribution sampling.

        Note: Assumes X is already preprocessed with categorical encoding
        handled by the base ImputerResults class.

        Args:
            X: Predictor matrix (already preprocessed).
            mean_quantile: Mean quantile for beta-distribution sampling. Only
                used when ``exact_quantile`` is None.
            count_samples: Number of samples for the legacy quantile-grid code
                path (kept for backward compatibility but no longer used by the
                unbiased implementation).
            exact_quantile: If provided, query the underlying QRF at exactly
                this quantile (no beta sampling). This guarantees monotonicity
                across quantiles for a given row and is used when the caller
                supplies an explicit ``quantiles`` list.
        """
        # Deterministic path: user asked for a specific quantile — query the
        # QRF directly so that for any row i,
        # prediction(q_low) <= prediction(q_mid) <= prediction(q_high).
        if exact_quantile is not None:
            pred = self.qrf.predict(X, quantiles=[float(exact_quantile)])
            pred = np.asarray(pred).reshape(len(X), -1)[:, 0]
            return pd.Series(pred, index=X.index, name=self.output_column)

        # Stochastic path: draw one continuous quantile per row from a Beta
        # distribution centred at ``mean_quantile`` (Beta(a,1) with
        # a = mean_quantile/(1 - mean_quantile); for mean_quantile=0.5 this
        # reduces to Uniform(0,1), so Beta(1,1) gives E[q]=0.5 and an
        # unbiased empirical median in the limit). The old implementation
        # converted a continuous beta draw into an integer index via
        # ``np.clip(beta(a,1)*10).astype(int)``, which *floored* (not
        # rounded) the index and used a grid of [0.091..0.909] — this
        # systematically biased the median low and truncated the tails.
        a = mean_quantile / (1 - mean_quantile)
        # Use the instance RNG so repeated predict() calls on the same X
        # produce independent draws (previously each call reset the RNG
        # from ``self.seed``, collapsing variance to zero).
        continuous_quantiles = self._rng.beta(a, 1, size=len(X))

        # Bucket continuous quantiles onto a fine symmetric grid covering the
        # full open interval (0, 1). Using round() (not floor) keeps the
        # mapping centred on the intended quantile, so the empirical mean of
        # mapped quantiles ≈ ``mean_quantile``. We avoid exact 0 and 1 because
        # QRF cannot extrapolate beyond observed extremes.
        grid_size = max(int(count_samples), 101)
        eps = 1.0 / (grid_size + 1)
        quantile_grid = np.linspace(eps, 1.0 - eps, grid_size)
        # Round (not floor) onto the grid to eliminate the low-side bias.
        grid_indices = np.clip(
            np.rint(continuous_quantiles * (grid_size - 1)).astype(int),
            0,
            grid_size - 1,
        )

        pred = self.qrf.predict(X, quantiles=list(quantile_grid))
        pred = np.asarray(pred)
        if pred.ndim == 2:
            predictions = pred[np.arange(len(X)), grid_indices]
        else:
            predictions = pred[np.arange(len(X)), :, grid_indices]

        return pd.Series(predictions, index=X.index, name=self.output_column)


class QRFResults(ImputerResults):
    """
    Fitted QRF instance ready for imputation.
    """

    def __init__(
        self,
        models: Dict[
            str, Any
        ],  # Can be _QRFModel, _RandomForestClassifierModel, or _ConstantValueModel
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
        """Initialize the QRF results.

        Args:
            models: Dictionary of fitted models (QRF or RF classifier) for each variable.
            predictors: List of column names used as predictors.
            imputed_variables: List of column names to be imputed.
            seed: Random seed for reproducibility.
            imputed_vars_dummy_info: Optional dictionary containing information
                about dummy variables for imputed variables.
            original_predictors: Optional list of original predictor variable
                names before dummy encoding.
            categorical_targets: Dictionary of categorical target info.
            boolean_targets: Dictionary of boolean target info.
            dummy_processor: Processor for handling dummy encoding in test data.
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

    @property
    def feature_names_in_(self) -> np.ndarray:
        """sklearn-style input feature names seen during fit.

        Uses the original (pre-dummy-encoding) predictor names when
        available, falling back to the encoded predictor list.
        """
        names = (
            self.original_predictors
            if getattr(self, "original_predictors", None)
            else self.predictors
        )
        return np.asarray(names, dtype=object)

    @property
    def feature_importances_(self):
        """sklearn-style feature importances from the underlying forest.

        For a single imputed variable, delegates to that variable's
        fitted ``RandomForestQuantileRegressor`` (``_QRFModel.qrf``). For
        multiple imputed variables, returns a ``{variable: importances}``
        dict. Raises ``AttributeError`` when no QRF-backed importances
        are available (e.g. constant or classifier models), so
        ``hasattr`` reports ``False``.
        """

        def _importances_for(variable: str) -> np.ndarray:
            model = self.models.get(variable)
            forest = getattr(model, "qrf", None)
            if forest is None:
                raise AttributeError(
                    "feature_importances_ is unavailable: model for "
                    f"{variable!r} has no underlying forest"
                )
            importances = getattr(forest, "feature_importances_", None)
            if importances is None:
                raise AttributeError(
                    "feature_importances_ is unavailable for " f"{variable!r}"
                )
            return importances

        if len(self.imputed_variables) == 1:
            return _importances_for(self.imputed_variables[0])
        return {var: _importances_for(var) for var in self.imputed_variables}

    def _get_encoded_predictors(
        self, current_predictors: List[str]
    ) -> List[str]:
        """Get properly encoded predictor columns for sequential imputation.

        Args:
            current_predictors: List of predictor variable names

        Returns:
            List of encoded predictor column names
        """
        if self.dummy_processor:
            return self.dummy_processor.get_sequential_predictor_columns(
                current_predictors
            )
        else:
            return current_predictors

    def _encode_imputed_variable(
        self, data: pd.DataFrame, variable: str
    ) -> pd.DataFrame:
        """Encode a categorical imputed variable for use as predictor in subsequent iterations.

        Args:
            data: DataFrame containing the imputed variable
            variable: Name of the variable that was just imputed

        Returns:
            DataFrame with encoded variable (adds dummy columns if categorical)
        """
        if (
            self.dummy_processor
            and variable in self.dummy_processor.imputed_var_dummy_mapping
        ):
            data = self.dummy_processor.sequential_imputed_predictor_encoding(
                data, variable
            )
            self.logger.debug(
                f"  Encoded '{variable}' for use in sequential imputation"
            )

        return data

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        mean_quantile: Optional[float] = 0.5,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict values at specified quantiles using the QRF model.

        Args:
            X_test: DataFrame containing the test data.
            quantiles: List of quantiles to predict (the quantile affects the
                center of the beta distribution from which to sample when imputing each data point).
            mean_quantile: The mean quantile to used for prediction if
                quantiles are not provided.
            return_probs: If True, return probability distributions for categorical variables.

        Returns:
            Dictionary mapping quantiles to predicted values.
            If return_probs=True, includes 'probabilities' key.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            # Create output dictionary with results
            imputations: Dict[float, pd.DataFrame] = {}
            prob_results = {} if return_probs else None

            # Convert single mean_quantile to a list if quantiles not provided
            quantiles_to_process = quantiles if quantiles else [mean_quantile]

            if quantiles:
                self.logger.info(
                    f"Predicting at {len(quantiles)} quantiles: {quantiles}"
                )
            else:
                self.logger.info(
                    f"Predicting from a beta distribution centered at quantile: {mean_quantile:.4f}"
                )

            for q in quantiles_to_process:
                imputed_df = pd.DataFrame()
                # Create a copy of X_test that we'll augment with imputed values
                X_test_augmented = X_test.copy()
                self.logger.debug(
                    f"X_test columns at start of _predict: {X_test_augmented.columns.tolist()}"
                )

                # Track dummy columns created from imputed categorical variables
                imputed_dummy_cols = set()

                for i, variable in enumerate(self.imputed_variables):
                    var_start_time = time.time()

                    if not quantiles:
                        self.logger.info(
                            f"[{i + 1}/{len(self.imputed_variables)}] Predicting for '{variable}'"
                        )

                    model = self.models[variable]

                    # Build predictor set: original predictors + previously imputed variables
                    var_predictors = _get_sequential_predictors(
                        self.predictors, self.imputed_variables, i
                    )

                    # Get properly encoded predictor columns
                    encoded_predictors = self._get_encoded_predictors(
                        var_predictors
                    )

                    self.logger.debug(
                        f"var_predictors for {variable}: {var_predictors}"
                    )
                    self.logger.debug(
                        f"encoded_predictors for {variable}: {encoded_predictors}"
                    )
                    self.logger.debug(
                        f"Available columns in X_test_augmented: {X_test_augmented.columns.tolist()}"
                    )

                    # Ensure we have all needed columns in X_test_augmented
                    missing_cols = set(encoded_predictors) - set(
                        X_test_augmented.columns
                    )
                    if missing_cols:
                        # Check if these are dummy columns from previously imputed categorical variables
                        imputed_missing = missing_cols & imputed_dummy_cols

                        if imputed_missing:
                            self.logger.debug(
                                f"Adding zero-filled columns for missing categories "
                                f"from imputed variables: {imputed_missing}"
                            )
                            # Add zeros for dummy columns from imputed categoricals
                            for col in imputed_missing:
                                X_test_augmented[col] = 0.0

                        # Any other missing columns will cause an error when we try to select them,
                        # which is the desired behavior to alert the user of missing predictors

                    # Import constant model
                    from microimpute.models.imputer import _ConstantValueModel

                    # Predict using the appropriate predictor set
                    if isinstance(model, _ConstantValueModel):
                        # Constant model - just return the constant value
                        imputed_values = model.predict(X_test_augmented)
                    elif isinstance(model, _RandomForestClassifierModel):
                        # Classification for categorical/boolean targets
                        if return_probs and prob_results is not None:
                            # Get probabilities and classes
                            prob_info = model.predict(
                                X_test_augmented[encoded_predictors],
                                return_probs=True,
                            )
                            prob_results[variable] = prob_info

                        # Get class predictions
                        imputed_values = model.predict(
                            X_test_augmented[encoded_predictors],
                            return_probs=False,
                        )
                    else:
                        # Regression for numeric targets.
                        # If the caller passed an explicit ``quantiles`` list
                        # (the user wants to inspect specific quantiles, e.g.
                        # for prediction intervals), we query the QRF at
                        # exactly ``q`` per row — NO beta sampling. This
                        # guarantees row-level monotonicity across quantiles.
                        # Otherwise, sample stochastically around ``q`` (the
                        # beta-mean default for imputation variance).
                        if quantiles:
                            imputed_values = model.predict(
                                X_test_augmented[encoded_predictors],
                                exact_quantile=q,
                            )
                        else:
                            imputed_values = model.predict(
                                X_test_augmented[encoded_predictors],
                                mean_quantile=q,
                            )

                    imputed_df[variable] = imputed_values

                    # Add the imputed values to X_test_augmented for subsequent variables
                    X_test_augmented[variable] = imputed_values

                    # Encode categorical/boolean imputed variable for next iteration
                    X_test_augmented = self._encode_imputed_variable(
                        X_test_augmented, variable
                    )

                    # Track the dummy columns that were added
                    if (
                        self.dummy_processor
                        and variable
                        in self.dummy_processor.imputed_var_dummy_mapping
                    ):
                        var_info = (
                            self.dummy_processor.imputed_var_dummy_mapping[
                                variable
                            ]
                        )
                        if var_info["dummy_cols"]:
                            imputed_dummy_cols.update(var_info["dummy_cols"])

                    # Log timing for individual variables when not processing multiple quantiles
                    if not quantiles:
                        var_time = time.time() - var_start_time
                        self.logger.info(
                            f"  ✓ {variable} predicted in {var_time:.2f}s ({len(imputed_values)} samples)"
                        )

                    self.logger.info(
                        f"QRF predictions completed for {variable} imputed variable"
                    )

                imputations[q] = imputed_df

            # Add probabilities to results if requested
            if return_probs and prob_results:
                imputations["probabilities"] = prob_results

            qs = [k for k in imputations.keys() if k != "probabilities"]
            if len(qs) < 2:
                q = list(qs)[0]

            # If quantiles not provided, decide what to return based on return_probs
            if not quantiles:
                if return_probs and prob_results:
                    # Return dict with both quantile predictions and probabilities
                    return imputations
                else:
                    # Return just the DataFrame for the single quantile
                    return imputations[q]
            else:
                # Multiple quantiles requested, return the full dict
                return imputations

        except Exception as e:
            self.logger.error(f"Error during QRF prediction: {str(e)}")
            raise RuntimeError(
                f"Failed to predict with QRF model: {str(e)}"
            ) from e


class QRF(BaseImputer):
    """
    Quantile Regression Forest model for imputation.

    This model uses a Quantile Regression Forest to predict quantiles.
    The underlying QRF implementation is from the quantile_forest package.
    """

    supports_target_filters = True

    def __init__(
        self,
        log_level: Optional[str] = "WARNING",
        memory_efficient: bool = False,
        batch_size: Optional[int] = None,
        cleanup_interval: int = 10,
        max_train_samples: Optional[int] = None,
    ) -> None:
        """Initialize the QRF model.

        Args:
            log_level: Logging level for the imputer.
            memory_efficient: Enable memory optimization features.
            batch_size: Process variables in batches to reduce memory usage.
            cleanup_interval: Frequency of garbage collection (every N variables).
            max_train_samples: If set, subsample X_train to at most this many
                rows before fitting. Reduces memory and training time while
                preserving sequential covariance structure.
        """
        super().__init__(log_level=log_level)
        self.models = {}
        self.log_level = log_level
        self.memory_efficient = memory_efficient
        self.batch_size = batch_size
        self.cleanup_interval = cleanup_interval
        if max_train_samples is not None and max_train_samples < 1:
            raise ValueError("max_train_samples must be a positive integer")
        self.max_train_samples = max_train_samples

        self.logger.debug("Initializing QRF imputer")

        if memory_efficient:
            self.logger.info(
                f"Memory-efficient mode enabled with cleanup_interval={cleanup_interval}"
            )
            if batch_size:
                self.logger.info(
                    f"Batch processing enabled with batch_size={batch_size}"
                )

    def _get_encoded_predictors(
        self,
        current_predictors: List[str],
        dummy_processor: Optional[Any] = None,
    ) -> List[str]:
        """Get properly encoded predictor columns for sequential imputation.

        This helper ensures consistent encoding of categorical predictors across
        all code paths (batch, non-batch, hyperparameter tuning, etc.).

        Args:
            current_predictors: List of predictor variable names
            dummy_processor: Optional DummyVariableProcessor instance

        Returns:
            List of encoded predictor column names
        """
        if dummy_processor is None:
            dummy_processor = getattr(self, "dummy_processor", None)

        if dummy_processor:
            return dummy_processor.get_sequential_predictor_columns(
                current_predictors
            )
        else:
            return current_predictors

    def _encode_imputed_variable(
        self,
        data: pd.DataFrame,
        variable: str,
        dummy_processor: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Encode a categorical imputed variable for use as predictor in subsequent iterations.

        This helper ensures consistent encoding of imputed categorical variables
        across all code paths.

        Args:
            data: DataFrame containing the imputed variable
            variable: Name of the variable that was just imputed
            dummy_processor: Optional DummyVariableProcessor instance

        Returns:
            DataFrame with encoded variable (adds dummy columns if categorical)
        """
        if dummy_processor is None:
            dummy_processor = getattr(self, "dummy_processor", None)

        if (
            dummy_processor
            and variable in dummy_processor.imputed_var_dummy_mapping
        ):
            data = dummy_processor.sequential_imputed_predictor_encoding(
                data, variable
            )
            self.logger.debug(
                f"  Encoded '{variable}' for use in sequential imputation"
            )

        return data

    def _create_model_for_variable(self, variable: str, **kwargs) -> Any:
        """Create the appropriate model (classifier or regressor) based on variable type."""
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        if variable in categorical_targets:
            # Use classifier for categorical targets
            return _RandomForestClassifierModel(
                seed=self.seed, logger=self.logger
            )
        elif variable in boolean_targets:
            # Use classifier for boolean targets
            return _RandomForestClassifierModel(
                seed=self.seed, logger=self.logger
            )
        else:
            # Use QRF for numeric targets
            return _QRFModel(seed=self.seed, logger=self.logger)

    def _fit_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        variable: str,
        **kwargs,
    ) -> None:
        """Fit the model with appropriate parameters based on variable type."""
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        # sample_weight is threaded via kwargs from the base BaseImputer.fit,
        # bypassing the nested qrf/rfc structure so both classifier and
        # regressor paths see the same per-row weights.
        sample_weight = kwargs.pop("sample_weight", None)

        # Extract appropriate parameters based on model type
        # Handle nested structure from hyperparameter tuning
        if isinstance(model, _RandomForestClassifierModel):
            # Use RFC params if they exist in a nested structure
            if "rfc" in kwargs:
                model_params = kwargs["rfc"]
            elif "qrf" in kwargs:
                # Mixed case: only QRF params available, use defaults for RFC
                model_params = {}
            else:
                # Flat dict: use all kwargs (backward compatible)
                model_params = kwargs

            if variable in categorical_targets:
                model.fit(
                    X,
                    y,
                    var_type=categorical_targets[variable]["type"],
                    categories=categorical_targets[variable].get("categories"),
                    sample_weight=sample_weight,
                    **model_params,
                )
            elif variable in boolean_targets:
                model.fit(
                    X,
                    y,
                    var_type="boolean",
                    sample_weight=sample_weight,
                    **model_params,
                )
        else:
            # Use QRF params if they exist in a nested structure
            if "qrf" in kwargs:
                model_params = kwargs["qrf"]
            elif "rfc" in kwargs:
                # Mixed case: only RFC params available, use defaults for QRF
                model_params = {}
            else:
                # Flat dict: use all kwargs (backward compatible)
                model_params = kwargs

            # Regular QRF fit
            model.fit(X, y, sample_weight=sample_weight, **model_params)

    def _target_fit_data(
        self,
        X_train: pd.DataFrame,
        variable: str,
        target_fit_masks: Optional[Dict[str, pd.Series]],
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Return training rows and weights for one target variable."""
        if not target_fit_masks or variable not in target_fit_masks:
            return X_train, sample_weight

        mask = (
            target_fit_masks[variable]
            .reindex(X_train.index)
            .fillna(False)
            .astype(bool)
        )
        if not mask.any():
            raise ValueError(
                f"No training rows selected for target '{variable}'"
            )

        target_train = X_train.loc[mask]
        target_sample_weight = None
        if sample_weight is not None:
            target_sample_weight = np.asarray(sample_weight, dtype=float)[
                mask.to_numpy()
            ]

        selected_rows = len(target_train)
        if (
            self.max_train_samples is not None
            and len(target_train) > self.max_train_samples
        ):
            try:
                variable_offset = (self.imputed_variables or []).index(
                    variable
                )
            except ValueError:
                variable_offset = 0
            seed = None if self.seed is None else self.seed + variable_offset
            rng = np.random.default_rng(seed)
            sel = rng.choice(
                len(target_train), size=self.max_train_samples, replace=False
            )
            target_train = target_train.iloc[sel]
            if target_sample_weight is not None:
                target_sample_weight = target_sample_weight[sel]
            self.logger.info(
                "Subsampling target '%s' training data from %d to %d rows",
                variable,
                selected_rows,
                self.max_train_samples,
            )

        dropped = len(X_train) - selected_rows
        if dropped:
            self.logger.info(
                "Target filter for '%s' selected %d/%d training rows",
                variable,
                selected_rows,
                len(X_train),
            )

        return target_train, target_sample_weight

    def _get_memory_usage_info(self) -> str:
        """Get formatted memory usage information."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        return "N/A"

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
        sample_weight: Optional[np.ndarray] = None,
        target_fit_masks: Optional[Dict[str, pd.Series]] = None,
        **qrf_kwargs: Any,
    ) -> QRFResults:
        """Fit the QRF model to the training data.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            sample_weight: Optional per-row sample weights threaded through
                to ``RandomForestQuantileRegressor.fit`` /
                ``RandomForestClassifier.fit``.
            **qrf_kwargs: Additional keyword arguments to pass to QRF.

        Returns:
            The fitted model instance.

        Raises:
            RuntimeError: If model fitting fails.
        """
        try:
            target_fit_masks = target_fit_masks or {}
            if tune_hyperparameters and target_fit_masks:
                raise NotImplementedError(
                    "QRF target_filters are not supported with tune_hyperparameters"
                )

            # Subsample training data if max_train_samples is set
            if (
                self.max_train_samples is not None
                and len(X_train) > self.max_train_samples
                and not target_fit_masks
            ):
                self.logger.info(
                    f"Subsampling training data from "
                    f"{len(X_train)} to {self.max_train_samples} rows"
                )
                # Sample by positional index so sample_weight stays aligned
                # with X_train after reset_index.
                rng = np.random.default_rng(self.seed)
                sel = rng.choice(
                    len(X_train), size=self.max_train_samples, replace=False
                )
                if sample_weight is not None:
                    sample_weight = np.asarray(sample_weight, dtype=float)[sel]
                X_train = X_train.iloc[sel].reset_index(drop=True)

            # Store target type information early for hyperparameter tuning
            self.categorical_targets = categorical_targets or {}
            self.boolean_targets = boolean_targets or {}
            self.numeric_targets = numeric_targets or []
            self.constant_targets = constant_targets or {}
            self.imputed_variables = imputed_variables

            if tune_hyperparameters:
                try:
                    qrf_kwargs = self._tune_hyperparameters(
                        data=X_train,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                    )

                    # Initialize and fit a QRF model for each variable
                    self.logger.info(
                        f"Training data shape: {X_train.shape}, Memory usage: {self._get_memory_usage_info()}"
                    )

                    # Handle batch processing if enabled
                    if (
                        self.batch_size
                        and len(imputed_variables) > self.batch_size
                    ):
                        self.logger.info(
                            f"Processing {len(imputed_variables)} variables in batches of {self.batch_size}"
                        )
                        variable_batches = [
                            imputed_variables[i : i + self.batch_size]
                            for i in range(
                                0, len(imputed_variables), self.batch_size
                            )
                        ]
                        for batch_idx, batch_variables in enumerate(
                            variable_batches
                        ):
                            self.logger.info(
                                f"Processing batch {batch_idx + 1}/{len(variable_batches)} "
                                f"({len(batch_variables)} variables)"
                            )
                            self._fit_variable_batch(
                                X_train,
                                predictors,
                                imputed_variables,
                                batch_variables,
                                qrf_kwargs,
                                constant_targets,
                                sample_weight=sample_weight,
                            )

                            # Memory cleanup after each batch
                            if self.memory_efficient:
                                gc.collect()
                                self.logger.info(
                                    f"Batch {batch_idx + 1} completed. Memory usage: {self._get_memory_usage_info()}"
                                )
                    else:
                        # Process all variables sequentially
                        # Import constant model
                        from microimpute.models.imputer import \
                            _ConstantValueModel

                        for i, variable in enumerate(imputed_variables):
                            var_start_time = time.time()

                            # Handle constant targets
                            if variable in (constant_targets or {}):
                                constant_val = constant_targets[variable][
                                    "value"
                                ]
                                self.models[variable] = _ConstantValueModel(
                                    constant_val, variable
                                )
                                self.logger.info(
                                    f"Using constant value {constant_val} for variable {variable}"
                                )
                                continue

                            # Build predictor set: original predictors + previously imputed variables
                            current_predictors = _get_sequential_predictors(
                                predictors, imputed_variables, i
                            )

                            # Get properly encoded predictor columns
                            dummy_processor = getattr(
                                self, "dummy_processor", None
                            )
                            encoded_predictors = self._get_encoded_predictors(
                                current_predictors, dummy_processor
                            )

                            # Log detailed pre-imputation information
                            self.logger.info(
                                f"[{i + 1}/{len(imputed_variables)}] Starting imputation for '{variable}'"
                            )
                            self.logger.info(
                                f"  Features: {len(encoded_predictors)} predictors"
                            )
                            self.logger.info(
                                f"  Memory usage: {self._get_memory_usage_info()}"
                            )

                            # Create appropriate model based on variable type
                            model = self._create_model_for_variable(variable)
                            target_train, target_sample_weight = (
                                self._target_fit_data(
                                    X_train,
                                    variable,
                                    target_fit_masks,
                                    sample_weight,
                                )
                            )
                            self._fit_model(
                                model,
                                target_train[encoded_predictors],
                                target_train[variable],
                                variable,
                                sample_weight=target_sample_weight,
                                **qrf_kwargs,
                            )

                            try:
                                # Log post-imputation information
                                var_time = time.time() - var_start_time
                                self.logger.info(
                                    f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                                )

                                # Get model complexity metrics if available
                                if hasattr(model, "qrf") and hasattr(
                                    model.qrf, "n_estimators"
                                ):
                                    self.logger.info(
                                        f"  Model complexity: {model.qrf.n_estimators} trees"
                                    )
                                elif hasattr(model, "classifier") and hasattr(
                                    model.classifier, "n_estimators"
                                ):
                                    self.logger.info(
                                        f"  Model complexity: {model.classifier.n_estimators} trees (classifier)"
                                    )

                                self.models[variable] = model

                                # Encode categorical/boolean imputed variable for next iteration
                                X_train = self._encode_imputed_variable(
                                    X_train, variable, dummy_processor
                                )

                            except Exception as e:
                                self.logger.error(
                                    f"  ✗ Failed: {variable} - {str(e)}"
                                )
                                raise

                            # Memory cleanup if enabled
                            if (
                                self.memory_efficient
                                and (i + 1) % self.cleanup_interval == 0
                            ):
                                gc.collect()
                                self.logger.debug(
                                    f"  Memory cleanup performed. Usage: {self._get_memory_usage_info()}"
                                )

                    return (
                        QRFResults(
                            models=self.models,
                            predictors=predictors,
                            imputed_variables=imputed_variables,
                            imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                            original_predictors=self.original_predictors,
                            categorical_targets=categorical_targets,
                            boolean_targets=boolean_targets,
                            constant_targets=constant_targets,
                            dummy_processor=getattr(
                                self, "dummy_processor", None
                            ),
                            seed=self.seed,
                        ),
                        qrf_kwargs,
                    )

                except Exception as e:
                    self.logger.error(
                        f"Error tuning hyperparameters: {str(e)}"
                    )
                    raise RuntimeError(
                        f"Failed to tune hyperparameters: {str(e)}"
                    ) from e

            else:
                self.logger.info(
                    f"Fitting QRF model with {len(predictors)} predictors and "
                    f"optional parameters: {qrf_kwargs}"
                )
                self.logger.info(
                    f"Training data shape: {X_train.shape}, Memory usage: {self._get_memory_usage_info()}"
                )

                # Handle batch processing if enabled
                if (
                    self.batch_size
                    and len(imputed_variables) > self.batch_size
                ):
                    self.logger.info(
                        f"Processing {len(imputed_variables)} variables in batches of {self.batch_size}"
                    )
                    variable_batches = [
                        imputed_variables[i : i + self.batch_size]
                        for i in range(
                            0, len(imputed_variables), self.batch_size
                        )
                    ]
                    for batch_idx, batch_variables in enumerate(
                        variable_batches
                    ):
                        self.logger.info(
                            f"Processing batch {batch_idx + 1}/{len(variable_batches)} "
                            f"({len(batch_variables)} variables)"
                        )
                        self._fit_variable_batch(
                            X_train,
                            predictors,
                            imputed_variables,
                            batch_variables,
                            qrf_kwargs,
                            constant_targets,
                            sample_weight=sample_weight,
                            target_fit_masks=target_fit_masks,
                        )

                        # Memory cleanup after each batch
                        if self.memory_efficient:
                            gc.collect()
                            self.logger.info(
                                f"Batch {batch_idx + 1} completed. Memory usage: {self._get_memory_usage_info()}"
                            )
                else:
                    # Process all variables sequentially
                    # Import constant model
                    from microimpute.models.imputer import _ConstantValueModel

                    # Initialize and fit a QRF model for each variable
                    for i, variable in enumerate(imputed_variables):
                        var_start_time = time.time()

                        # Handle constant targets
                        if variable in (constant_targets or {}):
                            constant_val = constant_targets[variable]["value"]
                            self.models[variable] = _ConstantValueModel(
                                constant_val, variable
                            )
                            self.logger.info(
                                f"Using constant value {constant_val} for variable {variable}"
                            )
                            continue

                        # Build predictor set: original predictors + previously imputed variables
                        current_predictors = _get_sequential_predictors(
                            predictors, imputed_variables, i
                        )

                        # Get properly encoded predictor columns
                        dummy_processor = getattr(
                            self, "dummy_processor", None
                        )
                        encoded_predictors = self._get_encoded_predictors(
                            current_predictors, dummy_processor
                        )

                        # Log detailed pre-imputation information
                        self.logger.info(
                            f"[{i + 1}/{len(imputed_variables)}] Starting imputation for '{variable}'"
                        )
                        self.logger.info(
                            f"  Features: {len(encoded_predictors)} predictors"
                        )
                        self.logger.info(
                            f"  Memory usage: {self._get_memory_usage_info()}"
                        )

                        # Create and fit model
                        model = self._create_model_for_variable(variable)

                        try:
                            target_train, target_sample_weight = (
                                self._target_fit_data(
                                    X_train,
                                    variable,
                                    target_fit_masks,
                                    sample_weight,
                                )
                            )
                            self._fit_model(
                                model,
                                target_train[encoded_predictors],
                                target_train[variable],
                                variable,
                                sample_weight=target_sample_weight,
                                **qrf_kwargs,
                            )

                            # Log post-imputation information
                            var_time = time.time() - var_start_time
                            self.logger.info(
                                f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                            )

                            # Get model complexity metrics if available
                            if hasattr(model, "qrf") and hasattr(
                                model.qrf, "n_estimators"
                            ):
                                self.logger.info(
                                    f"  Model complexity: {model.qrf.n_estimators} trees"
                                )
                            elif hasattr(model, "classifier") and hasattr(
                                model.classifier, "n_estimators"
                            ):
                                self.logger.info(
                                    f"  Model complexity: {model.classifier.n_estimators} trees (classifier)"
                                )

                            self.models[variable] = model

                            # Encode categorical/boolean imputed variable for next iteration
                            X_train = self._encode_imputed_variable(
                                X_train, variable, dummy_processor
                            )

                        except Exception as e:
                            self.logger.error(
                                f"  ✗ Failed: {variable} - {str(e)}"
                            )
                            raise

                        # Memory cleanup if enabled
                        if (
                            self.memory_efficient
                            and (i + 1) % self.cleanup_interval == 0
                        ):
                            gc.collect()
                            self.logger.debug(
                                f"  Memory cleanup performed. Usage: {self._get_memory_usage_info()}"
                            )

                # Final memory cleanup if enabled
                if self.memory_efficient:
                    gc.collect()

                self.logger.info(
                    f"QRF model fitting completed. Final memory usage: {self._get_memory_usage_info()}"
                )

                return QRFResults(
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
        except Exception as e:
            self.logger.error(f"Error fitting QRF model: {str(e)}")
            raise RuntimeError(f"Failed to fit QRF model: {str(e)}") from e

    def _fit_variable_batch(
        self,
        X_train: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        batch_variables: List[str],
        qrf_kwargs: Dict[str, Any],
        constant_targets: Optional[Dict[str, Dict]] = None,
        sample_weight: Optional[np.ndarray] = None,
        target_fit_masks: Optional[Dict[str, pd.Series]] = None,
    ) -> None:
        """Fit models for a batch of variables.

        Args:
            X_train: Training data
            predictors: Original predictor variables
            imputed_variables: All variables being imputed
            batch_variables: Variables in current batch
            qrf_kwargs: QRF model parameters
        """
        # Import constant model
        from microimpute.models.imputer import _ConstantValueModel

        for variable in batch_variables:
            var_start_time = time.time()
            i = imputed_variables.index(variable)

            # Handle constant targets
            if variable in (constant_targets or {}):
                constant_val = constant_targets[variable]["value"]
                self.models[variable] = _ConstantValueModel(
                    constant_val, variable
                )
                self.logger.info(
                    f"Using constant value {constant_val} for variable {variable}"
                )
                continue

            # Build predictor set: original predictors + previously imputed variables
            current_predictors = _get_sequential_predictors(
                predictors, imputed_variables, i
            )
            dummy_processor = getattr(self, "dummy_processor", None)
            encoded_predictors = self._get_encoded_predictors(
                current_predictors, dummy_processor
            )

            # Log detailed pre-imputation information
            self.logger.info(
                f"[{i + 1}/{len(imputed_variables)}] Starting imputation for '{variable}'"
            )
            self.logger.info(
                f"  Features: {len(encoded_predictors)} predictors"
            )
            self.logger.info(
                f"  Memory usage: {self._get_memory_usage_info()}"
            )

            # Create and fit model
            # Note: X_train is already preprocessed by base class
            model = self._create_model_for_variable(variable)

            try:
                target_train, target_sample_weight = self._target_fit_data(
                    X_train,
                    variable,
                    target_fit_masks,
                    sample_weight,
                )
                self._fit_model(
                    model,
                    target_train[encoded_predictors],
                    target_train[variable],
                    variable,
                    sample_weight=target_sample_weight,
                    **qrf_kwargs,
                )

                # Log post-imputation information
                var_time = time.time() - var_start_time
                self.logger.info(
                    f"  ✓ Success: {variable} fitted in {var_time:.2f}s"
                )

                # Get model complexity metrics if available
                if hasattr(model.qrf, "n_estimators"):
                    self.logger.info(
                        f"  Model complexity: {model.qrf.n_estimators} trees"
                    )

                self.models[variable] = model

            except Exception as e:
                self.logger.error(f"  ✗ Failed: {variable} - {str(e)}")
                raise

            # Memory cleanup if enabled
            if self.memory_efficient and (i + 1) % self.cleanup_interval == 0:
                gc.collect()
                self.logger.debug(
                    f"  Memory cleanup performed. Usage: {self._get_memory_usage_info()}"
                )

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the model and immediately predict, then release the fitted model.

        Convenience method that combines fit() + predict() + cleanup.
        Useful when you don't need to keep the fitted model around.

        Variables in ``imputed_variables`` that are missing from ``X_train``
        are automatically skipped during fitting and zero-filled in the
        output, so callers don't need to pre-filter.

        Args:
            X_train: DataFrame containing the training data.
            X_test: DataFrame containing the test data (predictors only).
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            **kwargs: Additional keyword arguments passed to fit().

        Returns:
            DataFrame with one column per imputed variable.
        """
        missing = [v for v in imputed_variables if v not in X_train.columns]
        if missing:
            self.logger.warning(
                f"fit_predict: {len(missing)} variables not in X_train "
                f"and will be zero-filled: {missing}"
            )

        fitted = self.fit(
            X_train=X_train,
            predictors=predictors,
            imputed_variables=imputed_variables,
            skip_missing=True,
            **kwargs,
        )

        result = fitted.predict(X_test=X_test[predictors])
        del fitted
        gc.collect()

        # Zero-fill missing variables to match the requested output shape.
        for var in missing:
            result[var] = 0

        # Reorder columns to match the original requested order.
        result = result[[v for v in imputed_variables if v in result.columns]]
        return result

    def _tune_qrf_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        numeric_vars: List[str],
        n_cv_folds: int = 3,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Tune hyperparameters for QRF model using quantile loss with CV.

        Args:
            data: Full training data.
            predictors: List of column names to use as predictors.
            numeric_vars: List of numeric variables to impute.
            n_cv_folds: Number of CV folds for robust evaluation (default: 3).
            n_trials: Number of Optuna trials (default: 10).

        Returns:
            Dictionary of tuned hyperparameters for QRF.
        """
        import optuna
        from sklearn.model_selection import KFold

        from microimpute.comparisons.metrics import compute_loss

        # Suppress Optuna's logs during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Get all imputed variables for proper sequential imputation
        all_imputed_vars = getattr(self, "imputed_variables", numeric_vars)

        # Set up CV folds
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.seed)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10
                ),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "bootstrap": trial.suggest_categorical(
                    "bootstrap", [True, False]
                ),
            }

            # Track errors across CV folds
            fold_errors = []

            # Perform CV
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data)):
                X_train_fold = data.iloc[train_idx]
                X_val_fold = data.iloc[val_idx]

                # Track errors for numeric variables in this fold
                var_errors = []

                # Create copies for augmented data
                X_train_augmented = X_train_fold.copy()
                X_val_augmented = X_val_fold.copy()

                # For each imputed variable (only evaluate numeric ones)
                for i, var in enumerate(all_imputed_vars):
                    # Build predictor set: original predictors + previously imputed variables
                    current_predictors = _get_sequential_predictors(
                        predictors, all_imputed_vars, i
                    )

                    # Get properly encoded predictor columns
                    dummy_processor = getattr(self, "dummy_processor", None)
                    encoded_predictors = self._get_encoded_predictors(
                        current_predictors, dummy_processor
                    )

                    # Only fit and evaluate numeric variables
                    if var in numeric_vars:
                        # Extract target variable values
                        y_val = X_val_fold[var]

                        # Create and fit QRF model with trial parameters
                        model = _QRFModel(seed=self.seed, logger=self.logger)
                        model.fit(
                            X_train_augmented[encoded_predictors],
                            X_train_fold[var],
                            **params,
                        )

                        # Predict
                        y_pred = model.predict(
                            X_val_augmented[encoded_predictors]
                        )

                        # Add predictions to augmented datasets for next variable
                        X_train_augmented[var] = model.predict(
                            X_train_augmented[encoded_predictors]
                        )
                        X_val_augmented[var] = y_pred

                        # Use quantile loss with median (q=0.5) for hyperparameter tuning
                        _, quantile_loss_value = compute_loss(
                            y_val.values.flatten(),
                            y_pred.values.flatten(),
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
                    else:
                        # Categorical variable - encode it for use as predictor in next iterations
                        if var in X_train_fold.columns:
                            X_train_augmented = self._encode_imputed_variable(
                                X_train_augmented, var, dummy_processor
                            )
                            X_val_augmented = self._encode_imputed_variable(
                                X_val_augmented, var, dummy_processor
                            )

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

        # Suppress warnings during optimization
        import os

        os.environ["PYTHONWARNINGS"] = "ignore"

        study.optimize(objective, n_trials=n_trials)

        best_value = study.best_value
        self.logger.info(
            f"QRF - Lowest average normalized quantile loss ({n_cv_folds}-fold CV): {best_value}"
        )

        best_params = study.best_params
        self.logger.info(f"QRF - Best hyperparameters found: {best_params}")

        return best_params

    def _tune_rfc_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        categorical_vars: List[str],
        n_cv_folds: int = 3,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Tune hyperparameters for RFC model using log loss with CV.

        Args:
            data: Full training data.
            predictors: List of column names to use as predictors.
            categorical_vars: List of categorical/boolean variables to impute.
            n_cv_folds: Number of CV folds for robust evaluation (default: 3).
            n_trials: Number of Optuna trials (default: 10).

        Returns:
            Dictionary of tuned hyperparameters for RFC.
        """
        import optuna
        from sklearn.model_selection import KFold

        from microimpute.comparisons.metrics import (
            compute_loss, order_probabilities_alphabetically)

        # Suppress Optuna's logs during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Get all imputed variables for proper sequential imputation
        all_imputed_vars = getattr(self, "imputed_variables", categorical_vars)
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        # Set up CV folds
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.seed)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
                ),
                "bootstrap": trial.suggest_categorical(
                    "bootstrap", [True, False]
                ),
            }

            # Track errors across CV folds
            fold_errors = []

            # Perform CV
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data)):
                X_train_fold = data.iloc[train_idx]
                X_val_fold = data.iloc[val_idx]

                # Track errors for categorical variables in this fold
                var_errors = []

                # Create copies for augmented data
                X_train_augmented = X_train_fold.copy()
                X_val_augmented = X_val_fold.copy()

                # For each imputed variable (only evaluate categorical ones)
                for i, var in enumerate(all_imputed_vars):
                    # Build predictor set: original predictors + previously imputed variables
                    current_predictors = _get_sequential_predictors(
                        predictors, all_imputed_vars, i
                    )

                    # Get properly encoded predictor columns
                    dummy_processor = getattr(self, "dummy_processor", None)
                    encoded_predictors = self._get_encoded_predictors(
                        current_predictors, dummy_processor
                    )

                    # Only fit and evaluate categorical/boolean variables
                    if var in categorical_vars:
                        # Extract target variable values
                        y_val = X_val_fold[var]

                        # Create and fit RFC model with trial parameters
                        model = _RandomForestClassifierModel(
                            seed=self.seed, logger=self.logger
                        )

                        # Determine variable type and fit appropriately
                        if var in categorical_targets:
                            model.fit(
                                X_train_augmented[encoded_predictors],
                                X_train_fold[var],
                                var_type=categorical_targets[var]["type"],
                                categories=categorical_targets[var].get(
                                    "categories"
                                ),
                                **params,
                            )
                        elif var in boolean_targets:
                            model.fit(
                                X_train_augmented[encoded_predictors],
                                X_train_fold[var],
                                var_type="boolean",
                                **params,
                            )

                        # Get probability predictions
                        prob_info = model.predict(
                            X_val_augmented[encoded_predictors],
                            return_probs=True,
                        )

                        # Get class predictions for augmented data
                        y_pred = model.predict(
                            X_val_augmented[encoded_predictors],
                            return_probs=False,
                        )

                        # Add predictions to augmented datasets for next variable
                        X_train_augmented[var] = model.predict(
                            X_train_augmented[encoded_predictors],
                            return_probs=False,
                        )
                        X_val_augmented[var] = y_pred

                        # Order probabilities alphabetically for log loss
                        probs_ordered, alphabetical_labels = (
                            order_probabilities_alphabetically(
                                prob_info["probabilities"],
                                prob_info["classes"],
                            )
                        )

                        # Compute log loss
                        _, log_loss_value = compute_loss(
                            y_val.values,
                            probs_ordered,
                            "log_loss",
                            labels=alphabetical_labels,
                        )

                        var_errors.append(log_loss_value)

                        # Encode the categorical variable for use as predictor in next iterations
                        X_train_augmented = self._encode_imputed_variable(
                            X_train_augmented, var, dummy_processor
                        )
                        X_val_augmented = self._encode_imputed_variable(
                            X_val_augmented, var, dummy_processor
                        )
                    else:
                        # Numeric variable - just add it to augmented data (already there from fold data)
                        pass

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

        # Suppress warnings during optimization
        import os

        os.environ["PYTHONWARNINGS"] = "ignore"

        study.optimize(objective, n_trials=n_trials)

        best_value = study.best_value
        self.logger.info(
            f"RFC - Lowest average log loss ({n_cv_folds}-fold CV): {best_value}"
        )

        best_params = study.best_params
        self.logger.info(f"RFC - Best hyperparameters found: {best_params}")

        return best_params

    @validate_call(config=VALIDATE_CONFIG)
    def _tune_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> Dict[str, Any]:
        """Tune hyperparameters for the QRF/RFC models using Optuna with CV.

        Automatically detects variable types and tunes appropriate models:
        - Numeric variables: QRF with quantile loss
        - Categorical/Boolean variables: RFC with log loss

        Uses cross-validation for robust hyperparameter selection.

        Args:
            data: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            Dictionary of tuned hyperparameters. Format depends on variable types:
            - Only numeric: flat dict with QRF params
            - Only categorical: flat dict with RFC params
            - Mixed: nested dict {"qrf": {...}, "rfc": {...}}
        """
        # Separate variables by type using existing class attributes
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        categorical_vars = [
            var
            for var in imputed_variables
            if var in categorical_targets or var in boolean_targets
        ]
        numeric_vars = [
            var for var in imputed_variables if var not in categorical_vars
        ]

        # Default: 3-fold CV with 10 trials (same computational cost as old 30 trials)
        n_cv_folds = 3
        n_trials = 10

        self.logger.info(
            f"Hyperparameter tuning with {n_cv_folds}-fold CV and {n_trials} trials: "
            f"{len(numeric_vars)} numeric variables, "
            f"{len(categorical_vars)} categorical/boolean variables"
        )

        # Tune appropriate models based on variable types
        if not categorical_vars:
            # Backward compatible: only numeric variables
            self.logger.info(
                "Tuning QRF hyperparameters (numeric variables only)"
            )
            return self._tune_qrf_hyperparameters(
                data, predictors, numeric_vars, n_cv_folds, n_trials
            )
        elif not numeric_vars:
            # Only categorical variables
            self.logger.info(
                "Tuning RFC hyperparameters (categorical/boolean variables only)"
            )
            return self._tune_rfc_hyperparameters(
                data, predictors, categorical_vars, n_cv_folds, n_trials
            )
        else:
            # Mixed: tune both separately
            self.logger.info(
                "Tuning both QRF and RFC hyperparameters (mixed variable types)"
            )
            qrf_params = self._tune_qrf_hyperparameters(
                data, predictors, numeric_vars, n_cv_folds, n_trials
            )
            rfc_params = self._tune_rfc_hyperparameters(
                data, predictors, categorical_vars, n_cv_folds, n_trials
            )
            return {"qrf": qrf_params, "rfc": rfc_params}
