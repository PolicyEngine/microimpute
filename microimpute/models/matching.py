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

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self, X_test: pd.DataFrame, quantiles: Optional[List[float]] = None
    ) -> Dict[float, pd.DataFrame]:
        """Predict imputed values using the matching model.

        Args:
            X_test: DataFrame containing the recipient data.
            quantiles: List of quantiles to predict.

        Returns:
            Dictionary mapping quantiles to imputed values.

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
                    X_test_copy, quantiles, chunk_size
                )
            else:
                return self._predict_single(X_test_copy, quantiles)

        except ValueError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error during matching prediction: {str(e)}")
            raise RuntimeError(f"Failed to perform matching: {str(e)}") from e

    def _predict_single(
        self,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
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

        return self._process_matching_results(fused0, X_test_copy, quantiles)

    def _predict_chunked(
        self,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]],
        chunk_size: int,
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
                combined_results, X_test_copy, quantiles
            )
        else:
            raise RuntimeError("No chunk results were produced")

    def _process_matching_results(
        self,
        fused0: pd.DataFrame,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]],
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

                    imputations[q] = imputed_df
            else:
                # If no quantiles specified, use a default one
                q = 0.5
                self.logger.info(
                    f"Creating imputation for default quantile {q}"
                )
                imputed_df = pd.DataFrame(index=X_test_copy.index)
                for variable in self.imputed_variables:
                    self.logger.info(f"Imputing variable {variable}")
                    imputed_df[variable] = fused0[variable].values
                imputations[q] = imputed_df

            # Verify output shapes
            for q, df in imputations.items():
                self.logger.debug(
                    f"Imputation result for q={q}: shape={df.shape}"
                )
                if len(df) != len(X_test_copy):
                    self.logger.warning(
                        f"Result shape mismatch: expected {len(X_test_copy)} rows, got {len(df)}"
                    )

            return imputations
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
        """Tune hyperparameters for the Matching model using Optuna.

        Args:
            X_train: DataFrame containing the training data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.

        Returns:
            Dictionary of tuned hyperparameters.
        """
        import optuna
        from sklearn.model_selection import train_test_split

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create a validation split (80% train, 20% validation)
        X_train, X_test = train_test_split(
            data, test_size=0.2, random_state=self.seed
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

            # Track errors for all variables
            var_errors = []

            for var in imputed_variables:
                y_test = X_test[var]
                X_test_var = X_test.copy().drop(var, axis=1)

                # Determine if chunking is needed for hyperparameter tuning
                chunk_size = 1000  # Smaller chunks for tuning
                total_size = len(X_train) * len(X_test_var)
                use_chunking = (
                    len(X_test_var) > chunk_size
                    or total_size > 25_000_000  # Lower threshold for tuning
                )

                if use_chunking:
                    # Perform chunked matching for hyperparameter tuning
                    y_pred_chunks = []
                    y_test_chunks = []

                    for i in range(0, len(X_test_var), chunk_size):
                        chunk_end = min(i + chunk_size, len(X_test_var))
                        chunk_data = X_test_var.iloc[i:chunk_end]
                        chunk_y_test = y_test.iloc[i:chunk_end]

                        try:
                            fused0, fused1 = self.matching_hotdeck(
                                receiver=chunk_data,
                                donor=X_train,
                                matching_variables=predictors,
                                z_variables=[var],
                                **params,
                            )
                            y_pred_chunks.append(fused0[var].values)
                            y_test_chunks.append(chunk_y_test.values)
                        except Exception:
                            # If chunk fails, use mean of training data as prediction
                            mean_val = X_train[var].mean()
                            y_pred_chunks.append(
                                np.full(len(chunk_data), mean_val)
                            )
                            y_test_chunks.append(chunk_y_test.values)

                    # Combine chunk results
                    y_pred = np.concatenate(y_pred_chunks)
                    y_test_combined = np.concatenate(y_test_chunks)
                else:
                    # Perform single matching
                    try:
                        fused0, fused1 = self.matching_hotdeck(
                            receiver=X_test_var,
                            donor=X_train,
                            matching_variables=predictors,
                            z_variables=[var],
                            **params,
                        )
                        y_pred = fused0[var].values
                        y_test_combined = y_test.values
                    except Exception:
                        # If matching fails, use mean of training data as prediction
                        mean_val = X_train[var].mean()
                        y_pred = np.full(len(X_test_var), mean_val)
                        y_test_combined = y_test.values

                # Calculate error
                # Normalize error by variable's standard deviation
                std = np.std(y_test_combined.flatten())
                mse = np.mean(
                    (y_pred.flatten() - y_test_combined.flatten()) ** 2
                )
                normalized_mse = mse / (std**2) if std > 0 else mse

                var_errors.append(normalized_mse)

            return np.mean(var_errors)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        # Suppress warnings during optimization
        import os

        os.environ["PYTHONWARNINGS"] = "ignore"

        study.optimize(objective, n_trials=30)

        best_value = study.best_value
        self.logger.info(f"Lowest average normalized MSE: {best_value}")

        best_params = study.best_params
        self.logger.info(f"Best hyperparameters found: {best_params}")

        return best_params
