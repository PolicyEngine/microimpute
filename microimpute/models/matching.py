"""Statistical matching imputation model using hot deck methods."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults
from microimpute.utils.statmatch_hotdeck import nnd_hotdeck_using_rpy2

MatchingHotdeckFn = Callable[
    [
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[List[str]],
        Optional[List[str]],
    ],
    Tuple[pd.DataFrame, pd.DataFrame],
]


class MatchingResults(ImputerResults):
    """
    Fitted Matching instance ready for imputation.
    """

    def __init__(
        self,
        matching_hotdeck: MatchingHotdeckFn,
        donor_data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        seed: int,
        imputed_vars_dummy_info: Optional[Dict[str, Any]] = None,
        original_predictors: Optional[List[str]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
        constant_targets: Optional[Dict[str, Dict]] = None,
        dummy_processor: Optional[Any] = None,
        log_level: Optional[str] = "WARNING",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the matching model.

        Args:
            matching_hotdeck: Function that performs the hot deck matching.
            donor_data: DataFrame containing the donor data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            seed: Random seed for reproducibility.
            imputed_vars_dummy_info: Optional dictionary containing information
                about dummy variables for imputed variables.
            original_predictors: Optional list of original predictor names
                before dummy encoding.
            categorical_targets: Dictionary of categorical target info.
            boolean_targets: Dictionary of boolean target info.
            dummy_processor: Processor for handling dummy encoding in test data.
            hyperparameters: Optional dictionary of hyperparameters for the
                matching function, specified after tunning.
        """
        super().__init__(
            predictors,
            imputed_variables,
            seed,
            imputed_vars_dummy_info,
            original_predictors,
            log_level,
        )
        self.matching_hotdeck = matching_hotdeck
        self.donor_data = donor_data
        self.hyperparameters = hyperparameters
        self.categorical_targets = categorical_targets or {}
        self.boolean_targets = boolean_targets or {}
        self.dummy_processor = dummy_processor

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values using the matching model.

        Args:
            X_test: DataFrame containing the recipient data.
            quantiles: List of quantiles to predict.
            return_probs: If True, return one-hot probability vectors for matched categories.

        Returns:
            Dictionary mapping quantiles to imputed values.
            If return_probs=True, includes 'probabilities' key with one-hot encodings.

        Raises:
            ValueError: If model is not properly set up or
                input data is invalid.
            RuntimeError: If matching or prediction fails.
        """
        try:
            self.logger.info(
                f"Performing matching for {len(X_test)} recipient records"
            )

            # Create a copy to avoid modifying the input
            try:
                self.logger.debug("Creating copy of test data")
                X_test_copy = X_test.copy()

                # Drop imputed variables if they exist in test data
                if any(
                    col in X_test.columns for col in self.imputed_variables
                ):
                    self.logger.debug(
                        f"Dropping imputed variables from test data: {self.imputed_variables}"
                    )
                    X_test_copy.drop(
                        self.imputed_variables,
                        axis=1,
                        inplace=True,
                        errors="ignore",
                    )
            except Exception as copy_error:
                self.logger.error(
                    f"Error preparing test data: {str(copy_error)}"
                )
                raise RuntimeError(
                    "Failed to prepare test data for matching"
                ) from copy_error

            # Determine if chunking is needed for large datasets
            chunk_size = 2000
            total_size = len(self.donor_data) * len(X_test_copy)
            use_chunking = (
                len(X_test_copy) > chunk_size
                or total_size > 50_000_000  # 50M combinations threshold
            )

            if use_chunking:
                self.logger.info(
                    f"Large dataset detected ({len(X_test_copy)} receiver records, "
                    f"{len(self.donor_data)} donor records). Using chunking approach."
                )
                return self._predict_chunked(
                    X_test_copy, quantiles, chunk_size, return_probs
                )
            else:
                return self._predict_single(
                    X_test_copy, quantiles, return_probs
                )

        except ValueError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error during matching prediction: {str(e)}")
            raise RuntimeError(f"Failed to perform matching: {str(e)}") from e

    def _predict_single(
        self,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Perform matching on the full dataset without chunking."""
        try:
            self.logger.info("Calling R-based hot deck matching function")
            if self.hyperparameters:
                fused0, fused1 = self.matching_hotdeck(
                    receiver=X_test_copy,
                    donor=self.donor_data,
                    matching_variables=self.predictors,
                    z_variables=self.imputed_variables,
                    **self.hyperparameters,
                )
            else:
                fused0, fused1 = self.matching_hotdeck(
                    receiver=X_test_copy,
                    donor=self.donor_data,
                    matching_variables=self.predictors,
                    z_variables=self.imputed_variables,
                )
        except Exception as matching_error:
            self.logger.error(
                f"Error in hot deck matching: {str(matching_error)}"
            )
            raise RuntimeError("Hot deck matching failed") from matching_error

        return self._process_matching_results(
            fused0, X_test_copy, quantiles, return_probs
        )

    def _predict_chunked(
        self,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]],
        chunk_size: int,
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Perform matching using chunking for large datasets."""
        all_results = []

        # Process receiver data in chunks
        for i in range(0, len(X_test_copy), chunk_size):
            chunk_end = min(i + chunk_size, len(X_test_copy))
            chunk_data = X_test_copy.iloc[i:chunk_end]

            self.logger.debug(
                f"Processing chunk {i//chunk_size + 1}: "
                f"rows {i} to {chunk_end-1} ({len(chunk_data)} records)"
            )

            try:
                # Perform matching for this chunk
                if self.hyperparameters:
                    fused0, fused1 = self.matching_hotdeck(
                        receiver=chunk_data,
                        donor=self.donor_data,
                        matching_variables=self.predictors,
                        z_variables=self.imputed_variables,
                        **self.hyperparameters,
                    )
                else:
                    fused0, fused1 = self.matching_hotdeck(
                        receiver=chunk_data,
                        donor=self.donor_data,
                        matching_variables=self.predictors,
                        z_variables=self.imputed_variables,
                    )

                # Store results with original indices
                chunk_results = pd.DataFrame(index=chunk_data.index)
                for variable in self.imputed_variables:
                    chunk_results[variable] = fused0[variable].values

                all_results.append(chunk_results)

            except Exception as chunk_error:
                self.logger.warning(
                    f"Chunk {i//chunk_size + 1} failed: {chunk_error}. "
                    "Filling with NaN values."
                )
                # Create NaN-filled results for failed chunk
                chunk_results = pd.DataFrame(index=chunk_data.index)
                for variable in self.imputed_variables:
                    chunk_results[variable] = np.nan
                all_results.append(chunk_results)

        # Combine all chunk results, preserving original order
        if all_results:
            combined_results = pd.concat(all_results)
            combined_results = combined_results.loc[X_test_copy.index]

            return self._process_matching_results(
                combined_results, X_test_copy, quantiles, return_probs
            )
        else:
            raise RuntimeError("No chunk results were produced")

    def _generate_one_hot_probabilities(
        self,
        variable: str,
        matched_values: np.ndarray,
        index: pd.Index,
        categorical_targets: Dict,
        boolean_targets: Dict,
    ) -> Optional[Dict]:
        """Generate one-hot probability matrix for categorical/boolean variables.

        Args:
            variable: Name of the variable
            matched_values: Array of matched category values
            index: Index for the output DataFrame
            categorical_targets: Dictionary of categorical target info
            boolean_targets: Dictionary of boolean target info

        Returns:
            Dict with 'probabilities' and 'classes' keys
        """
        if (
            variable not in categorical_targets
            and variable not in boolean_targets
        ):
            return None

        # Determine categories
        if variable in boolean_targets:
            categories = [False, True]
        else:
            categories = categorical_targets[variable].get("categories", [])

        if not categories:
            return None

        # Create probability matrix (one-hot encoding)
        n_samples = len(matched_values)
        n_categories = len(categories)
        prob_matrix = np.zeros((n_samples, n_categories))

        # Set 1.0 for matched category
        for idx, val in enumerate(matched_values):
            try:
                cat_idx = categories.index(val)
                prob_matrix[idx, cat_idx] = 1.0
            except ValueError:
                # If value not found in categories, default to first category
                prob_matrix[idx, 0] = 1.0

        return {"probabilities": prob_matrix, "classes": np.array(categories)}

    def _process_matching_results(
        self,
        fused0: pd.DataFrame,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]],
        return_probs: bool = False,
    ) -> Dict[float, pd.DataFrame]:
        """Process matching results into the expected output format."""
        try:
            # Verify imputed variables exist in the result
            missing_imputed = [
                var
                for var in self.imputed_variables
                if var not in fused0.columns
            ]
            if missing_imputed:
                self.logger.error(
                    f"Imputed variables missing from matching result: {missing_imputed}"
                )
                raise ValueError(
                    f"Matching failed to produce these variables: {missing_imputed}"
                )

            self.logger.info(
                f"Matching completed, fused dataset has {len(fused0)} records"
            )
        except Exception as convert_error:
            self.logger.error(
                f"Error converting matching results: {str(convert_error)}"
            )
            raise RuntimeError(
                "Failed to process matching results"
            ) from convert_error

        # Create output dictionary with results
        imputations: Dict[float, pd.DataFrame] = {}
        prob_results = {} if return_probs else None

        # Get target type information if available
        categorical_targets = getattr(self, "categorical_targets", {})
        boolean_targets = getattr(self, "boolean_targets", {})

        try:
            if quantiles:
                self.logger.info(
                    f"Creating imputations for {len(quantiles)} quantiles"
                )
                # For each quantile, return a DataFrame with all imputed variables
                for q in quantiles:
                    imputed_df = pd.DataFrame(index=X_test_copy.index)
                    for variable in self.imputed_variables:
                        self.logger.debug(
                            f"Adding result for imputed variable {variable} at quantile {q}"
                        )
                        imputed_df[variable] = fused0[variable].values

                        # Generate one-hot probabilities if requested
                        if return_probs and prob_results is not None:
                            prob_df = self._generate_one_hot_probabilities(
                                variable,
                                fused0[variable].values,
                                X_test_copy.index,
                                categorical_targets,
                                boolean_targets,
                            )
                            if prob_df is not None:
                                prob_results[variable] = prob_df

                    imputations[q] = imputed_df

                # Add probabilities to results if requested
                if return_probs and prob_results:
                    imputations["probabilities"] = prob_results

                return imputations
            else:
                # If no quantiles specified, use a default one
                q_default = 0.5
                self.logger.info(
                    f"Creating imputation for default quantile {q_default}"
                )
                imputed_df = pd.DataFrame(index=X_test_copy.index)
                for variable in self.imputed_variables:
                    self.logger.info(f"Imputing variable {variable}")
                    imputed_df[variable] = fused0[variable].values

                    # Generate one-hot probabilities if requested
                    if return_probs and prob_results is not None:
                        prob_df = self._generate_one_hot_probabilities(
                            variable,
                            fused0[variable].values,
                            X_test_copy.index,
                            categorical_targets,
                            boolean_targets,
                        )
                        if prob_df is not None:
                            prob_results[variable] = prob_df

                imputations[q_default] = imputed_df

                # Add probabilities to results if requested
                if return_probs and prob_results:
                    # Return dict with both quantile predictions and probabilities
                    imputations["probabilities"] = prob_results
                    return imputations
                else:
                    # Return just the DataFrame for the single quantile
                    return imputations[q_default]
        except Exception as output_error:
            self.logger.error(
                f"Error creating output imputations: {str(output_error)}"
            )
            raise RuntimeError(
                "Failed to create output imputations"
            ) from output_error


class Matching(Imputer):
    """
    Statistical matching model for imputation using nearest neighbor distance
    hot deck method.

    This model uses R's StatMatch package through rpy2 to perform nearest
    neighbor distance hot deck matching for imputation.
    """

    def __init__(
        self,
        matching_hotdeck: MatchingHotdeckFn = nnd_hotdeck_using_rpy2,
        log_level: Optional[str] = "WARNING",
    ) -> None:
        """Initialize the matching model.

        Args:
            matching_hotdeck: Function that performs the hot deck matching.
            log_level: Logging level for the model.

        Raises:
            ValueError: If matching_hotdeck is not callable
        """
        super().__init__(log_level=log_level)
        self.log_level = log_level
        self.logger.debug("Initializing Matching imputer")

        # Validate input
        if not callable(matching_hotdeck):
            self.logger.error("matching_hotdeck must be a callable function")
            raise ValueError("matching_hotdeck must be a callable function")

        self.matching_hotdeck = matching_hotdeck
        self.donor_data: Optional[pd.DataFrame] = None

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
        **matching_kwargs: Any,
    ) -> MatchingResults:
        """Fit the matching model by storing the donor data and variable names.

        Args:
            X_train: DataFrame containing the donor data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            matching_kwargs: Additional keyword arguments for hyperparameter
                tuning of the matching function.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If matching cannot be set up.
        """
        try:
            self.donor_data = X_train.copy()

            if tune_hyperparameters:
                self.logger.info(
                    "Tuning hyperparameters for the matching model"
                )
                best_params = self._tune_hyperparameters(
                    data=X_train,
                    predictors=predictors,
                    imputed_variables=imputed_variables,
                )
                self.logger.info(f"Best hyperparameters: {best_params}")

                return (
                    MatchingResults(
                        matching_hotdeck=self.matching_hotdeck,
                        donor_data=self.donor_data,
                        predictors=predictors,
                        imputed_variables=imputed_variables,
                        imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                        original_predictors=self.original_predictors,
                        seed=self.seed,
                        hyperparameters=best_params,
                    ),
                    best_params,
                )

            else:
                self.logger.info(
                    f"Matching model ready with {len(X_train)} donor records and "
                    f"optional parameters: {matching_kwargs}"
                )
                self.logger.info(f"Using predictors: {predictors}")
                self.logger.info(
                    f"Targeting imputed variables: {imputed_variables}"
                )

                return MatchingResults(
                    matching_hotdeck=self.matching_hotdeck,
                    donor_data=self.donor_data,
                    predictors=predictors,
                    imputed_variables=imputed_variables,
                    imputed_vars_dummy_info=self.imputed_vars_dummy_info,
                    original_predictors=self.original_predictors,
                    categorical_targets=categorical_targets,
                    boolean_targets=boolean_targets,
                    dummy_processor=getattr(self, "dummy_processor", None),
                    seed=self.seed,
                    log_level=self.log_level,
                    hyperparameters=matching_kwargs,
                )
        except Exception as e:
            self.logger.error(f"Error setting up matching model: {str(e)}")
            raise ValueError(
                f"Failed to set up matching model: {str(e)}"
            ) from e

    @validate_call(config=VALIDATE_CONFIG)
    def _tune_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
    ) -> Dict[str, Any]:
        """Tune hyperparameters for the Matching model using Optuna with CV.

        Uses cross-validation and quantile loss for robust hyperparameter selection.

        Args:
            data: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            Dictionary of tuned hyperparameters.
        """
        import optuna
        from sklearn.model_selection import KFold

        from microimpute.comparisons.metrics import compute_loss

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Use 3-fold CV with 10 trials
        n_cv_folds = 3
        n_trials = 10

        # Set up CV folds
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.seed)

        self.logger.info(
            f"Tuning Matching hyperparameters with {n_cv_folds}-fold CV and {n_trials} trials"
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "dist_fun": trial.suggest_categorical(
                    "dist_fun",
                    [
                        "Manhattan",
                        "Euclidean",
                        "Mahalanobis",
                        "Gower",
                        "minimax",
                    ],
                ),
                "constrained": trial.suggest_categorical(
                    "constrained", [False, True]
                ),
                "constr_alg": trial.suggest_categorical(
                    "constr_alg", ["hungarian", "lpSolve"]
                ),
                "k": trial.suggest_int("k", 1, 10),
            }

            # Detect variable types for appropriate metric selection
            from microimpute.comparisons.metrics import (
                get_metric_for_variable_type,
            )

            variable_metrics = {}
            for var in imputed_variables:
                variable_metrics[var] = get_metric_for_variable_type(
                    data[var], var
                )

            # Track errors across CV folds
            fold_errors = []

            # Perform CV
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data)):
                X_train_fold = data.iloc[train_idx]
                X_val_fold = data.iloc[val_idx]

                # Track errors for all variables in this fold
                var_errors = []

                for var in imputed_variables:
                    y_val = X_val_fold[var]
                    X_val_var = X_val_fold.copy().drop(var, axis=1)

                    # Determine if chunking is needed for hyperparameter tuning
                    chunk_size = 1000  # Smaller chunks for tuning
                    total_size = len(X_train_fold) * len(X_val_var)
                    use_chunking = (
                        len(X_val_var) > chunk_size
                        or total_size
                        > 25_000_000  # Lower threshold for tuning
                    )

                    if use_chunking:
                        # Perform chunked matching for hyperparameter tuning
                        y_pred_chunks = []
                        y_val_chunks = []

                        for i in range(0, len(X_val_var), chunk_size):
                            chunk_end = min(i + chunk_size, len(X_val_var))
                            chunk_data = X_val_var.iloc[i:chunk_end]
                            chunk_y_val = y_val.iloc[i:chunk_end]

                            try:
                                fused0, fused1 = self.matching_hotdeck(
                                    receiver=chunk_data,
                                    donor=X_train_fold,
                                    matching_variables=predictors,
                                    z_variables=[var],
                                    **params,
                                )
                                y_pred_chunks.append(fused0[var].values)
                                y_val_chunks.append(chunk_y_val.values)
                            except Exception:
                                # If chunk fails, use mean of training data as prediction
                                mean_val = X_train_fold[var].mean()
                                y_pred_chunks.append(
                                    np.full(len(chunk_data), mean_val)
                                )
                                y_val_chunks.append(chunk_y_val.values)

                        # Combine chunk results
                        y_pred = np.concatenate(y_pred_chunks)
                        y_val_combined = np.concatenate(y_val_chunks)
                    else:
                        # Perform single matching
                        try:
                            fused0, fused1 = self.matching_hotdeck(
                                receiver=X_val_var,
                                donor=X_train_fold,
                                matching_variables=predictors,
                                z_variables=[var],
                                **params,
                            )
                            y_pred = fused0[var].values
                            y_val_combined = y_val.values
                        except Exception:
                            # If matching fails, use mean of training data as prediction
                            mean_val = X_train_fold[var].mean()
                            y_pred = np.full(len(X_val_var), mean_val)
                            y_val_combined = y_val.values

                    # Use appropriate metric based on variable type
                    metric = variable_metrics[var]

                    if metric == "quantile_loss":
                        _, loss_value = compute_loss(
                            y_val_combined.flatten(),
                            y_pred.flatten(),
                            "quantile_loss",
                            q=0.5,
                        )
                        # Normalize by variable's standard deviation
                        std = np.std(y_val_combined.flatten())
                        normalized_loss = (
                            loss_value / std if std > 0 else loss_value
                        )
                    else:  # log_loss for categorical/boolean
                        _, loss_value = compute_loss(
                            y_val_combined.flatten(),
                            y_pred.flatten(),
                            "log_loss",
                        )
                        # Log loss is already normalized
                        normalized_loss = loss_value

                    var_errors.append(normalized_loss)

                # Average across variables for this fold
                if var_errors:
                    fold_errors.append(np.mean(var_errors))

            # Return mean error across all CV folds
            return np.mean(fold_errors) if fold_errors else float("inf")

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
            f"Matching - Lowest average normalized quantile loss ({n_cv_folds}-fold CV): {best_value}"
        )

        best_params = study.best_params
        self.logger.info(
            f"Matching - Best hyperparameters found: {best_params}"
        )

        return best_params
