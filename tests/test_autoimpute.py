"""
Test the autoimpute function.
"""

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.autoimpute import autoimpute
from microimpute.visualizations.plotting import *


def test_autoimpute_basic() -> None:
    """Test that autoimpute returns expected data structures."""
    diabetes = load_diabetes()
    diabetes_donor = pd.DataFrame(
        diabetes.data, columns=diabetes.feature_names
    )
    # Add random boolean variable
    diabetes_donor["bool"] = np.random.choice(
        [True, False], size=len(diabetes_donor)
    )
    diabetes_receiver = pd.DataFrame(
        diabetes.data, columns=diabetes.feature_names
    )

    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "bool"]

    results = autoimpute(
        donor_data=diabetes_donor,
        receiver_data=diabetes_receiver,
        predictors=predictors,
        imputed_variables=imputed_variables,
        hyperparameters={
            "QRF": {"n_estimators": 100},
            "Matching": {"constrained": True},
        },
        log_level="INFO",
    )

    model_imputations = results.imputations
    imputed_data = results.receiver_data
    method_results_df = results.cv_results

    # Check that the imputations is a dictionary of dataframes
    assert isinstance(model_imputations, dict)
    for model, imputations in model_imputations.items():
        assert isinstance(imputations, pd.DataFrame)

    # Check that the method_results_df has the expected structure
    assert isinstance(method_results_df, pd.DataFrame)
    # method_results_df will have quantiles as columns and model names as indices
    assert "mean_loss" in method_results_df.columns
    assert 0.05 in method_results_df.columns  # First quantile
    assert 0.95 in method_results_df.columns  # Last quantile

    quantiles = [q for q in method_results_df.columns if isinstance(q, float)]

    model_imputations["best_method"].to_csv(
        "autoimpute_bestmodel_median_imputations.csv"
    )
    imputed_data.to_csv("autoimpute_bestmodel_imputed_dataset.csv")

    method_results_df.to_csv("autoimpute_model_comparison_results.csv")

    comparison_viz = method_comparison_results(
        data=method_results_df,
        metric_name="Test Quantile Loss",
        data_format="wide",  # Explicitly using wide format
    )
    fig = comparison_viz.plot(
        title="Autoimpute Method Comparison",
        show_mean=True,
        save_path="autoimpute_model_comparison.jpg",
    )
