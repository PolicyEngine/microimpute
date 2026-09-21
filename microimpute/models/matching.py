"""Statistical matching imputation model using hot deck methods."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import validate_call

from microimpute.config import RANDOM_STATE, VALIDATE_CONFIG
from microimpute.models.imputer import Imputer, ImputerResults


def nnd_hotdeck_using_rpy2(*args, **kwargs):
    """Load the optional R bridge only when the default matcher is called."""
    from microimpute.utils.statmatch_hotdeck import nnd_hotdeck_using_rpy2 as match

    return match(*args, **kwargs)


nnd_hotdeck_using_rpy2._microimpute_seeded_adapter = True


def _is_seeded_adapter(matching_hotdeck: Callable) -> bool:
    """Recognize native bridges without importing the optional R runtime."""
    return bool(getattr(matching_hotdeck, "_microimpute_seeded_adapter", False))


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
        self._rng = np.random.default_rng(seed)
        self.categorical_targets = categorical_targets or {}
        self.boolean_targets = boolean_targets or {}
        self.dummy_processor = dummy_processor
        self.n_failed_records = 0

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore the donor-draw stream when loading historical results."""
        self.__dict__.update(state)
        if "_rng" not in state:
            self._rng = np.random.default_rng(self.seed)

    def _matching_kwargs(self) -> Dict[str, Any]:
        """Advance a reproducible child-seed stream for the optional R bridge."""
        kwargs = dict(self.hyperparameters or {})
        if _is_seeded_adapter(self.matching_hotdeck):
            kwargs["random_state"] = int(self._rng.integers(0, np.iinfo(np.int32).max))
        return kwargs

    @validate_call(config=VALIDATE_CONFIG)
    def _predict(
        self,
        X_test: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        return_probs: bool = False,
    ) -> pd.DataFrame:
        """Predict imputed values using the matching model.

        Args:
            X_test: DataFrame containing the recipient data.
            quantiles: Unsupported; Matching returns donor draws.
            return_probs: Unsupported; Matching does not estimate probabilities.

        Returns:
            DataFrame of donor draws, with n_failed_records in its attrs.

        Side effects:
            Sets ``self.n_failed_records`` to the number of recipient records
            that could not be matched and are NaN in the result, and mirrors it
            on ``result.attrs["n_failed_records"]``. It is reset to 0 on entry,
            so it always describes the most recent call. Matching runs
            single-threaded (``autoimpute`` forces ``n_jobs=1`` when a Matching
            model is present), so concurrent calls on one fitted object would
            race on it.

        Raises:
            ValueError: If model is not properly set up or
                input data is invalid.
            RuntimeError: If matching or prediction fails.
            NotImplementedError: If quantiles or probabilities are requested.
        """
        # Clear the previous call's failure count before validating this request.
        self.n_failed_records = 0
        if quantiles is not None:
            raise NotImplementedError(
                "Matching returns donor draws, not conditional quantiles. "
                "Call predict without quantiles, or use QRF, OLS, or QuantReg."
            )
        if return_probs:
            raise NotImplementedError(
                "Matching does not estimate class probabilities. Use QRF or OLS."
            )
        try:
            self.logger.info(f"Performing matching for {len(X_test)} recipient records")

            # Create a copy to avoid modifying the input
            try:
                self.logger.debug("Creating copy of test data")
                X_test_copy = X_test.copy()

                # Drop imputed variables if they exist in test data
                if any(col in X_test.columns for col in self.imputed_variables):
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
                self.logger.error(f"Error preparing test data: {str(copy_error)}")
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
                return self._predict_single(X_test_copy, quantiles, return_probs)

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
    ) -> pd.DataFrame:
        """Perform matching on the full dataset without chunking."""
        try:
            self.logger.info("Calling R-based hot deck matching function")
            fused0, fused1 = self.matching_hotdeck(
                receiver=X_test_copy,
                donor=self.donor_data,
                matching_variables=self.predictors,
                z_variables=self.imputed_variables,
                **self._matching_kwargs(),
            )
        except Exception as matching_error:
            self.logger.error(f"Error in hot deck matching: {str(matching_error)}")
            raise RuntimeError("Hot deck matching failed") from matching_error

        self.n_failed_records = int(
            fused0[self.imputed_variables].isna().any(axis=1).sum()
        )
        return self._process_matching_results(
            fused0, X_test_copy, quantiles, return_probs
        )

    def _predict_chunked(
        self,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]],
        chunk_size: int,
        return_probs: bool = False,
    ) -> pd.DataFrame:
        """Perform matching using chunking for large datasets."""
        all_results = []

        # Process receiver data in chunks
        for i in range(0, len(X_test_copy), chunk_size):
            chunk_end = min(i + chunk_size, len(X_test_copy))
            chunk_data = X_test_copy.iloc[i:chunk_end]

            self.logger.debug(
                f"Processing chunk {i // chunk_size + 1}: "
                f"rows {i} to {chunk_end - 1} ({len(chunk_data)} records)"
            )

            try:
                # Perform matching for this chunk
                fused0, fused1 = self.matching_hotdeck(
                    receiver=chunk_data,
                    donor=self.donor_data,
                    matching_variables=self.predictors,
                    z_variables=self.imputed_variables,
                    **self._matching_kwargs(),
                )

                # Store results with original indices
                chunk_results = pd.DataFrame(index=chunk_data.index)
                for variable in self.imputed_variables:
                    chunk_results[variable] = fused0[variable].values

                all_results.append(chunk_results)

            except Exception as chunk_error:
                self.logger.warning(
                    f"Chunk {i // chunk_size + 1} failed: {chunk_error}. "
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

            # A failed chunk leaves NaN blocks in the output. Report the total
            # so a caller knows what share of the result is missing without
            # having to check for it themselves.
            n_failed = int(combined_results.isna().any(axis=1).sum())
            if n_failed:
                self.logger.warning(
                    f"{n_failed} of {len(combined_results)} records "
                    f"({n_failed / len(combined_results):.1%}) could not be "
                    "matched and are NaN in the result."
                )
            self.n_failed_records = n_failed

            return self._process_matching_results(
                combined_results, X_test_copy, quantiles, return_probs
            )
        else:
            raise RuntimeError("No chunk results were produced")

    def _process_matching_results(
        self,
        fused0: pd.DataFrame,
        X_test_copy: pd.DataFrame,
        quantiles: Optional[List[float]],
        return_probs: bool = False,
    ) -> pd.DataFrame:
        """Return donor draws, with an explicit count of unsuccessful matches."""
        if quantiles is not None:
            raise NotImplementedError(
                "Matching does not estimate conditional quantiles"
            )
        if return_probs:
            raise NotImplementedError("Matching does not estimate class probabilities")
        missing = [v for v in self.imputed_variables if v not in fused0]
        if missing:
            raise ValueError(f"Matching failed to produce these variables: {missing}")
        if len(fused0) != len(X_test_copy):
            raise ValueError("Matching must return one record per receiver")
        output = pd.DataFrame(
            {v: fused0[v].to_numpy() for v in self.imputed_variables},
            index=X_test_copy.index,
        )
        self.n_failed_records = int(output.isna().any(axis=1).sum())
        output.attrs["n_failed_records"] = self.n_failed_records
        if self.n_failed_records:
            self.logger.warning(
                f"{self.n_failed_records} of {len(output)} records "
                f"({self.n_failed_records / len(output):.1%}) could not be "
                "matched and are NaN in the result."
            )
        return output


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
        seed: int = RANDOM_STATE,
    ) -> None:
        """Initialize the matching model.

        Args:
            matching_hotdeck: Function that performs the hot deck matching.
            log_level: Logging level for the model.

        Raises:
            ValueError: If matching_hotdeck is not callable
        """
        super().__init__(seed=seed, log_level=log_level)
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
        sample_weight: Optional[np.ndarray] = None,
        **matching_kwargs: Any,
    ) -> MatchingResults:
        """Fit the matching model by storing the donor data and variable names.

        Args:
            X_train: DataFrame containing the donor data.
            predictors: List of column names to use as predictors.
            imputed_variables: List of column names to impute.
            sample_weight: Optional per-row sample weights for the donor
                dataset. When provided, weights are passed to R StatMatch's
                ``RANDwNND.hotdeck`` via a donor weight column. By default,
                donors tied at the minimum distance are sampled in proportion
                to their weights.
            matching_kwargs: Additional keyword arguments for hyperparameter
                tuning of the matching function.

        Returns:
            The fitted model instance.

        Raises:
            ValueError: If matching cannot be set up.
        """
        try:
            self.donor_data = X_train.copy()
            if sample_weight is not None:
                # Attach donor weights to the matching hyperparameters so
                # they're forwarded into the StatMatch R call (weight.don).
                matching_kwargs = {
                    **matching_kwargs,
                    "donor_sample_weight": np.asarray(sample_weight, dtype=float),
                }

            if tune_hyperparameters:
                self.logger.info("Tuning hyperparameters for the matching model")
                best_params = self._tune_hyperparameters(
                    data=X_train,
                    predictors=predictors,
                    imputed_variables=imputed_variables,
                    matching_kwargs=matching_kwargs,
                    categorical_targets=categorical_targets,
                    boolean_targets=boolean_targets,
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
                        categorical_targets=categorical_targets,
                        boolean_targets=boolean_targets,
                        dummy_processor=getattr(self, "dummy_processor", None),
                        seed=self.seed,
                        log_level=self.log_level,
                        hyperparameters={**matching_kwargs, **best_params},
                    ),
                    best_params,
                )

            else:
                self.logger.info(
                    f"Matching model ready with {len(X_train)} donor records and "
                    f"optional parameters: {matching_kwargs}"
                )
                self.logger.info(f"Using predictors: {predictors}")
                self.logger.info(f"Targeting imputed variables: {imputed_variables}")

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
            raise ValueError(f"Failed to set up matching model: {str(e)}") from e

    @validate_call(config=VALIDATE_CONFIG)
    def _tune_hyperparameters(
        self,
        data: pd.DataFrame,
        predictors: List[str],
        imputed_variables: List[str],
        matching_kwargs: Optional[Dict[str, Any]] = None,
        categorical_targets: Optional[Dict[str, Dict]] = None,
        boolean_targets: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """Tune donor-draw accuracy; failed matches prune the entire trial.

        Numeric donor draws are assessed with absolute error normalized by the
        donor training standard deviation; categorical draws use misclassification
        rate. These are internal tuning criteria, not predictive distribution scores.
        """
        import optuna
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        fixed_kwargs = dict(matching_kwargs or {})
        weights = fixed_kwargs.pop("donor_sample_weight", None)
        discrete_targets = set(categorical_targets or {}) | set(boolean_targets or {})

        # Keeps the most recent underlying failure so an all-pruned study can
        # report why, rather than only that nothing succeeded.
        last_trial_error: Optional[BaseException] = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal last_trial_error
            # NND.hotdeck's k controls donor re-use only under constrained
            # matching; it is not a nearest-neighbor count. Do not tune a no-op.
            params = {
                "dist_fun": trial.suggest_categorical(
                    "dist_fun",
                    ["Manhattan", "Euclidean", "Mahalanobis", "Gower", "minimax"],
                )
            }
            fold_errors = []
            # Common random numbers across trials make parameter comparisons
            # reproducible without rewarding a different random donor sequence.
            trial_rng = np.random.default_rng(self.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data)):
                donor = data.iloc[train_idx]
                receiver = data.iloc[val_idx].drop(columns=imputed_variables)
                call_kwargs = {**fixed_kwargs, **params}
                if weights is not None:
                    call_kwargs["donor_sample_weight"] = np.asarray(weights)[train_idx]
                predicted = []
                for start in range(0, len(receiver), 1000):
                    chunk = receiver.iloc[start : start + 1000]
                    if _is_seeded_adapter(self.matching_hotdeck):
                        call_kwargs["random_state"] = int(
                            trial_rng.integers(0, np.iinfo(np.int32).max)
                        )
                    try:
                        fused, _ = self.matching_hotdeck(
                            receiver=chunk,
                            donor=donor,
                            matching_variables=predictors,
                            z_variables=imputed_variables,
                            **call_kwargs,
                        )
                        if (
                            len(fused) != len(chunk)
                            or fused[imputed_variables].isna().any().any()
                        ):
                            raise ValueError("Matching returned incomplete predictions")
                        predicted.append(
                            fused[imputed_variables].reset_index(drop=True)
                        )
                    except Exception as error:
                        last_trial_error = error
                        self.logger.warning(
                            f"Matching failed on fold {fold_idx} chunk {start}: {error}. Pruning trial."
                        )
                        raise optuna.TrialPruned() from error
                predictions = pd.concat(predicted, ignore_index=True)
                errors = []
                for var in imputed_variables:
                    actual = data.iloc[val_idx][var].to_numpy()
                    estimate = predictions[var].to_numpy()
                    if var in discrete_targets:
                        errors.append(float(np.mean(actual != estimate)))
                    else:
                        actual = actual.astype(float)
                        estimate = estimate.astype(float)
                        if not np.isfinite(estimate).all():
                            raise optuna.TrialPruned(
                                "Matching returned nonfinite numeric predictions"
                            )
                        scale = float(donor[var].std(ddof=0))
                        errors.append(
                            float(np.mean(np.abs(actual - estimate))) / (scale or 1.0)
                        )
                fold_errors.append(float(np.mean(errors)))
            return float(np.mean(fold_errors))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=10)
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            message = "No matching hyperparameter trial succeeded"
            if last_trial_error is not None:
                message += f". Last error: {last_trial_error}"
            raise ValueError(message)
        self.logger.info(
            f"Matching best normalized donor-draw error: {study.best_value}"
        )
        return study.best_params
