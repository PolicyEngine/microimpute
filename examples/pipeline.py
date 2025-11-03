"""
Comprehensive pipeline demonstrating the full functionality of microimpute.

This script demonstrates:
1. Running autoimpute on the diabetes dataset
2. Evaluating distribution preservation
3. Analyzing predictor correlations and mutual information
4. Assessing predictor importance via leave-one-out analysis
5. Analyzing impact of variable ordering via progressive inclusion
6. Formatting all results into a unified CSV for dashboard visualization
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_diabetes

# Import microimpute functions
from microimpute.comparisons.autoimpute import autoimpute
from microimpute.comparisons.metrics import compare_distributions
from microimpute.evaluations.predictor_analysis import (
    compute_predictor_correlations,
    leave_one_out_analysis,
    progressive_predictor_inclusion,
)
from microimpute.utils.dashboard_formatter import format_csv

# Import model classes
from microimpute.models import OLS

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_full_pipeline(output_path="microimpute_results.csv"):
    """
    Run the complete microimpute pipeline on the diabetes dataset.

    Parameters
    ----------
    output_path : str
        Path to save the formatted CSV output

    Returns
    -------
    pd.DataFrame
        Formatted results ready for dashboard visualization
    """
    print("=" * 80)
    print("MICROIMPUTE COMPREHENSIVE PIPELINE DEMONSTRATION")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Load and prepare data
    # ========================================================================
    print("STEP 1: Loading and preparing diabetes dataset...")
    print("-" * 80)

    # Load the diabetes dataset
    diabetes = load_diabetes()
    diabetes_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # Split into donor and receiver portions (70/30 split)
    np.random.seed(42)
    donor_indices = np.random.choice(
        len(diabetes_data),
        size=int(0.7 * len(diabetes_data)),
        replace=False,
    )
    receiver_indices = np.array(
        [i for i in range(len(diabetes_data)) if i not in donor_indices]
    )

    donor_data = diabetes_data.iloc[donor_indices].reset_index(drop=True)
    receiver_data = diabetes_data.iloc[receiver_indices].reset_index(drop=True)

    # Create a categorical risk_factor variable based on cholesterol levels (s4)
    # Categorize into low, medium, high based on s4 values
    def categorize_risk(s4_value):
        if s4_value < -0.02:
            return "low"
        elif s4_value < 0.02:
            return "medium"
        else:
            return "high"

    donor_data["risk_factor"] = donor_data["s4"].apply(categorize_risk)
    receiver_data["risk_factor"] = receiver_data["s4"].apply(categorize_risk)

    # Define predictors and variables to impute
    predictors = ["age", "sex", "bmi", "bp"]
    imputed_variables = ["s1", "s4", "risk_factor"]

    # Remove imputed variables from receiver data
    receiver_data_without_targets = receiver_data.drop(
        columns=imputed_variables
    )

    print(f"Donor data shape: {donor_data.shape}")
    print(f"Receiver data shape: {receiver_data_without_targets.shape}")
    print(f"Predictors: {predictors}")
    print(f"Variables to impute: {imputed_variables}")
    print(f"Risk factor distribution in donor data:")
    print(donor_data["risk_factor"].value_counts())
    print()

    # ========================================================================
    # STEP 2: Run autoimpute
    # ========================================================================
    print("STEP 2: Running autoimpute...")
    print("-" * 80)

    autoimpute_results = autoimpute(
        donor_data=donor_data,
        receiver_data=receiver_data_without_targets,
        predictors=predictors,
        imputed_variables=imputed_variables,
        tune_hyperparameters=False,
        impute_all=True,  # Get imputations from all methods
        k_folds=3,
    )

    # Get the best method name
    best_method_name = autoimpute_results.fitted_models[
        "best_method"
    ].__class__.__name__
    print(f"Best performing method: {best_method_name}")
    print()

    # ========================================================================
    # STEP 3: Evaluate distribution preservation
    # ========================================================================
    print("STEP 3: Evaluating distribution preservation...")
    print("-" * 80)

    # For distribution comparison, we compare donor data with receiver data
    # that now has imputed values
    distribution_comparison_df = compare_distributions(
        donor_data=donor_data,
        receiver_data=autoimpute_results.receiver_data,
        imputed_variables=imputed_variables,
    )

    print("Distribution comparison results:")
    print(distribution_comparison_df)
    print()

    # ========================================================================
    # STEP 4: Analyze predictor correlations and mutual information
    # ========================================================================
    print("STEP 4: Computing predictor correlations and mutual information...")
    print("-" * 80)

    predictor_correlations = compute_predictor_correlations(
        data=donor_data,
        predictors=predictors,
        imputed_variables=imputed_variables,
        method="all",  # Compute all correlation types
    )

    print("Correlation analysis completed:")
    print(
        f"  - Pearson correlation matrix: {predictor_correlations['pearson'].shape}"
    )
    print(
        f"  - Spearman correlation matrix: {predictor_correlations['spearman'].shape}"
    )
    print(
        f"  - Mutual information matrix: {predictor_correlations['mutual_info'].shape}"
    )
    print(
        f"  - Predictor-target MI: {predictor_correlations['predictor_target_mi'].shape}"
    )
    print()

    # ========================================================================
    # STEP 5: Assess predictor importance (leave-one-out analysis)
    # ========================================================================
    print("STEP 5: Performing leave-one-out predictor importance analysis...")
    print("-" * 80)

    predictor_importance_df = leave_one_out_analysis(
        data=donor_data,
        predictors=predictors,
        imputed_variables=imputed_variables,
        model_class=OLS,  # Use the same model class for consistency
        quantiles=[0.1, 0.5, 0.9],
        train_size=0.7,
        n_jobs=1,
        random_state=42,
    )

    print("Predictor importance results:")
    print(predictor_importance_df[["predictor_removed", "relative_impact"]])
    print()

    # ========================================================================
    # STEP 6: Analyze impact of variable ordering
    # ========================================================================
    print("STEP 6: Analyzing impact of variable ordering...")
    print("-" * 80)

    progressive_results = progressive_predictor_inclusion(
        data=donor_data,
        predictors=predictors,
        imputed_variables=imputed_variables,
        model_class=OLS,
        quantiles=[0.1, 0.5, 0.9],
        train_size=0.7,
        random_state=42,
    )

    # Extract results
    progressive_inclusion_df = progressive_results["results_df"]
    optimal_subset = progressive_results["optimal_subset"]
    optimal_loss = progressive_results["optimal_loss"]

    print(
        f"Optimal predictor order: {progressive_inclusion_df['predictor_added'].tolist()}"
    )
    print(f"Optimal subset: {optimal_subset}")
    print(f"Optimal loss: {optimal_loss:.6f}")
    print()

    # ========================================================================
    # STEP 7: Format all results into unified CSV
    # ========================================================================
    print("STEP 7: Formatting results for dashboard visualization...")
    print("-" * 80)

    # Convert autoimpute_results to dictionary format expected by format_csv
    autoimpute_dict = {"cv_results": autoimpute_results.cv_results}

    # Format all results
    formatted_df = format_csv(
        output_path=output_path,
        autoimpute_result=autoimpute_dict,
        comparison_metrics_df=None,  # cv_results already contain this info
        distribution_comparison_df=distribution_comparison_df,
        predictor_correlations=predictor_correlations,
        predictor_importance_df=predictor_importance_df,
        progressive_inclusion_df=progressive_inclusion_df,
        best_method_name=best_method_name,
        donor_data=donor_data,
        receiver_data=autoimpute_results.receiver_data,
        imputed_variables=imputed_variables,
    )

    print(f"Formatted DataFrame shape: {formatted_df.shape}")
    print(f"Result types included: {formatted_df['type'].unique()}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Summary of results:")
    print(f"  - Total rows in output: {len(formatted_df)}")
    print(f"  - Best imputation method: {best_method_name}")
    print(f"  - Number of predictors analyzed: {len(predictors)}")
    print(f"  - Number of imputed variables: {len(imputed_variables)}")
    print(f"    - Numerical variables: s1, s4")
    print(f"    - Categorical variables: risk_factor")
    print()
    print("Output CSV contains:")
    for result_type in formatted_df["type"].unique():
        count = len(formatted_df[formatted_df["type"] == result_type])
        print(f"  - {result_type}: {count} rows")
    print()
    print(
        f"The output file '{output_path}' is ready for dashboard visualization."
    )
    print()

    return formatted_df


if __name__ == "__main__":
    # Run the full pipeline
    results_df = run_full_pipeline(
        output_path="microimputation-dashboard/public/microimputation_results.csv"
    )
