"""
OpenML AutoML Benchmark Regression Suite Data Loader

This module retrieves and analyzes datasets from the OpenML AutoML
Benchmark Regression suite (ID: 269) for evaluating statistical
matching models in microimpute.
"""

import re

import openml
import pandas as pd

# Patterns that indicate generic/non-interpretable feature names
GENERIC_FEATURE_PATTERNS = [
    r"^col_?\d+$",  # col_1, col1, col_2
    r"^[fvx]\d+$",  # f1, v1, x1, V1, X1
    r"^feat(ure)?_?\d+$",  # feat1, feature_1
    r"^var_?\d+$",  # var1, var_1
    r"^att(r)?_?\d+$",  # attr1, att_1
    r"^[a-z]\d+[a-z]*\d*$",  # P1, P1p2, H2p2 (encoded names)
    r"^[a-f0-9]{8,}$",  # Hashed/anonymized names (e.g., 48df886f9)
]


def is_generic_feature_name(name: str) -> bool:
    """
    Check if a feature name appears to be generic/non-interpretable.

    Parameters
    ----------
    name : str
        Feature name to check

    Returns
    -------
    bool
        True if the name matches a generic pattern
    """
    name_lower = name.lower().strip()
    for pattern in GENERIC_FEATURE_PATTERNS:
        if re.match(pattern, name_lower):
            return True
    return False


def calculate_interpretability_score(feature_names: list) -> float:
    """
    Calculate the proportion of features with interpretable names.

    Parameters
    ----------
    feature_names : list
        List of feature names

    Returns
    -------
    float
        Proportion of features with non-generic names (0 to 1)
    """
    if not feature_names:
        return 0.0

    interpretable_count = sum(
        1 for name in feature_names if not is_generic_feature_name(name)
    )
    return interpretable_count / len(feature_names)


def list_all_qualities(dataset_id: int = 269) -> list:
    """
    List all available qualities for a sample dataset to understand
    what metadata OpenML provides.

    Parameters
    ----------
    dataset_id : int
        Sample dataset ID to inspect qualities

    Returns
    -------
    list
        List of all available quality names
    """
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=False,
        download_qualities=True,
    )
    if dataset.qualities:
        return sorted(dataset.qualities.keys())
    return []


def get_benchmark_suite_metadata(suite_id: int = 269) -> pd.DataFrame:
    """
    Retrieve metadata for all datasets in an OpenML benchmark suite.

    Parameters
    ----------
    suite_id : int
        The OpenML benchmark suite ID (default: 269 for AutoML
        Benchmark Regression)

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata for each dataset in the suite
    """
    # Get the benchmark suite
    suite = openml.study.get_suite(suite_id)
    print(f"Suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Number of datasets: {len(suite.data)}")
    print("-" * 80)

    # Collect metadata for each dataset
    metadata_list = []

    for dataset_id in suite.data:
        try:
            dataset = openml.datasets.get_dataset(
                dataset_id,
                download_data=False,
                download_qualities=True,
                download_features_meta_data=True,
            )

            # Get qualities (dataset characteristics)
            qualities = dataset.qualities if dataset.qualities else {}

            # Get feature information
            features = dataset.features if dataset.features else {}
            feature_names = []
            feature_types = {}
            if features:
                for feat in features.values():
                    feature_names.append(feat.name)
                    feat_type = feat.data_type
                    feature_types[feat_type] = feature_types.get(feat_type, 0) + 1

            # Calculate interpretability score
            interpretability = calculate_interpretability_score(feature_names)

            metadata = {
                "dataset_id": dataset_id,
                "name": dataset.name,
                "version": dataset.version,
                "n_instances": qualities.get("NumberOfInstances"),
                "n_features": qualities.get("NumberOfFeatures"),
                "n_numeric_features": qualities.get("NumberOfNumericFeatures"),
                "n_categorical_features": qualities.get("NumberOfSymbolicFeatures"),
                "n_missing_values": qualities.get("NumberOfMissingValues"),
                "pct_missing": qualities.get("PercentageOfMissingValues"),
                "n_instances_with_missing": qualities.get(
                    "NumberOfInstancesWithMissingValues"
                ),
                # Relationship/correlation measures
                "mean_mutual_info": qualities.get("MeanMutualInformation"),
                "mean_attr_entropy": qualities.get("MeanAttributeEntropy"),
                "equiv_num_attr": qualities.get("EquivalentNumberOfAtts"),
                "noise_signal_ratio": qualities.get("NoiseToSignalRatio"),
                "class_entropy": qualities.get("ClassEntropy"),
                "mean_kurtosis": qualities.get("MeanKurtosisOfNumericAtts"),
                "mean_skewness": qualities.get("MeanSkewnessOfNumericAtts"),
                "target_variable": dataset.default_target_attribute,
                "interpretability_score": interpretability,
                "feature_names": feature_names,
                "description": (
                    dataset.description[:200] + "..."
                    if dataset.description and len(dataset.description) > 200
                    else dataset.description
                ),
                "format": dataset.format,
                "upload_date": dataset.upload_date,
            }

            metadata_list.append(metadata)
            print(f"Loaded: {dataset.name} (ID: {dataset_id})")

        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {e}")
            metadata_list.append(
                {
                    "dataset_id": dataset_id,
                    "name": "ERROR",
                    "error": str(e),
                }
            )

    return pd.DataFrame(metadata_list)


def analyze_missingness(df_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missingness patterns in the benchmark datasets.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        DataFrame with dataset metadata

    Returns
    -------
    pd.DataFrame
        Summary of datasets with missing values
    """
    missing_df = df_metadata[df_metadata["n_missing_values"] > 0].copy()
    missing_df = missing_df.sort_values("pct_missing", ascending=False)
    return missing_df[
        [
            "name",
            "n_instances",
            "n_features",
            "n_missing_values",
            "pct_missing",
            "n_instances_with_missing",
        ]
    ]


def filter_datasets(
    df_metadata: pd.DataFrame,
    min_numeric_proportion: float = 0.5,
    max_missing_values: int = 0,
    max_instances: int = 40000,
    min_instances: int = 0,
    min_features: int = 0,
    max_features: int = None,
    exclude_name_patterns: list = None,
    min_interpretability_score: float = None,
) -> pd.DataFrame:
    """
    Filter datasets based on specified criteria.

    Parameters
    ----------
    df_metadata : pd.DataFrame
        DataFrame with dataset metadata from get_benchmark_suite_metadata()
    min_numeric_proportion : float
        Minimum proportion of features that must be numeric (default: 0.5)
    max_missing_values : int
        Maximum number of missing values allowed (default: 0 for complete)
    max_instances : int
        Maximum number of instances (default: 40000)
    min_instances : int
        Minimum number of instances (default: 0)
    min_features : int
        Minimum number of features (default: 0)
    max_features : int
        Maximum number of features (default: None for no limit)
    exclude_name_patterns : list
        List of substrings to exclude from dataset names (default: None)
        Case-insensitive matching.
    min_interpretability_score : float
        Minimum proportion of features with interpretable names (default:
        None for no filtering). Value between 0 and 1, where 1 means all
        features must have interpretable names.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only datasets meeting all criteria
    """
    df = df_metadata.copy()

    # Calculate numeric proportion
    df["numeric_proportion"] = df["n_numeric_features"] / df["n_features"]

    # Apply filters
    mask = (
        (df["numeric_proportion"] >= min_numeric_proportion)
        & (df["n_missing_values"] <= max_missing_values)
        & (df["n_instances"] <= max_instances)
        & (df["n_instances"] >= min_instances)
        & (df["n_features"] >= min_features)
    )

    if max_features is not None:
        mask = mask & (df["n_features"] <= max_features)

    # Filter by interpretability score
    if min_interpretability_score is not None:
        mask = mask & (df["interpretability_score"] >= min_interpretability_score)

    # Exclude datasets by name pattern
    if exclude_name_patterns:
        for pattern in exclude_name_patterns:
            mask = mask & (~df["name"].str.lower().str.contains(pattern.lower()))

    filtered_df = df[mask].copy()

    return filtered_df


def save_filtered_dataset_ids(
    df_filtered: pd.DataFrame,
    output_path: str = "paper/benchmarking_datasets/selected_dataset_ids.txt",
) -> list:
    """
    Save filtered dataset IDs to a file.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered DataFrame from filter_datasets()
    output_path : str
        Path to save the dataset IDs

    Returns
    -------
    list
        List of selected dataset IDs
    """
    dataset_ids = df_filtered["dataset_id"].tolist()

    with open(output_path, "w") as f:
        f.write("# Selected OpenML Dataset IDs\n")
        f.write("# Filtering criteria applied - see data_loader.py\n")
        for did in dataset_ids:
            f.write(f"{int(did)}\n")

    return dataset_ids


def download_filtered_datasets(
    df_filtered: pd.DataFrame,
    output_dir: str = "paper/benchmarking_datasets/datasets",
    file_format: str = "csv",
) -> dict:
    """
    Download all datasets that passed filtering criteria.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        Filtered DataFrame from filter_datasets()
    output_dir : str
        Directory to save downloaded datasets
    file_format : str
        Output format: 'csv' or 'parquet' (default: 'csv')

    Returns
    -------
    dict
        Dictionary with dataset names as keys and info dict as values
        containing 'path', 'shape', 'target', 'features'
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    downloaded = {}
    total = len(df_filtered)

    for _, row in df_filtered.iterrows():
        dataset_id = int(row["dataset_id"])
        dataset_name = row["name"]

        print(
            f"[{len(downloaded) + 1}/{total}] Downloading {dataset_name} "
            f"(ID: {dataset_id})..."
        )

        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )

            # Combine features and target into single DataFrame
            df = X.copy()
            target_name = dataset.default_target_attribute
            if target_name:
                df[target_name] = y

            # Clean filename
            clean_name = dataset_name.lower().replace(" ", "_").replace("-", "_")

            # Save to file
            if file_format == "parquet":
                output_path = os.path.join(output_dir, f"{clean_name}.parquet")
                df.to_parquet(output_path, index=False)
            else:
                output_path = os.path.join(output_dir, f"{clean_name}.csv")
                df.to_csv(output_path, index=False)

            downloaded[dataset_name] = {
                "dataset_id": dataset_id,
                "path": output_path,
                "shape": df.shape,
                "target": target_name,
                "features": list(X.columns),
                "categorical_features": (
                    [
                        name
                        for name, is_cat in zip(attribute_names, categorical_indicator)
                        if is_cat
                    ]
                    if categorical_indicator
                    else []
                ),
            }

            print(f"    Saved: {output_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

        except Exception as e:
            print(f"    ERROR: {e}")
            downloaded[dataset_name] = {"error": str(e)}

    # Save summary
    summary_path = os.path.join(output_dir, "download_summary.csv")
    summary_data = []
    for name, info in downloaded.items():
        if "error" not in info:
            summary_data.append(
                {
                    "name": name,
                    "dataset_id": info["dataset_id"],
                    "path": info["path"],
                    "n_rows": info["shape"][0],
                    "n_cols": info["shape"][1],
                    "target": info["target"],
                    "n_categorical": len(info["categorical_features"]),
                }
            )
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Successfully downloaded: {len(summary_data)}/{total} datasets")

    return downloaded


def get_dataset_details(dataset_id: int) -> dict:
    """
    Get detailed information about a specific dataset including
    feature relationships.

    Parameters
    ----------
    dataset_id : int
        OpenML dataset ID

    Returns
    -------
    dict
        Detailed dataset information
    """
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )

    X, _, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    details = {
        "name": dataset.name,
        "description": dataset.description,
        "target": dataset.default_target_attribute,
        "features": attribute_names,
        "categorical_features": [
            name
            for name, is_cat in zip(attribute_names, categorical_indicator)
            if is_cat
        ],
        "numeric_features": [
            name
            for name, is_cat in zip(attribute_names, categorical_indicator)
            if not is_cat
        ],
        "shape": X.shape,
        "missing_per_feature": X.isnull().sum().to_dict(),
        "data_sample": X.head(),
    }

    return details


if __name__ == "__main__":
    # Use first dataset in suite to discover available qualities
    suite = openml.study.get_suite(269)
    first_dataset_id = suite.data[0]
    all_qualities = list_all_qualities(first_dataset_id)

    # Fetch and display benchmark suite metadata
    print("\n" + "=" * 80)
    print("OpenML AutoML Benchmark Regression Suite Analysis")
    print("=" * 80)

    metadata_df = get_benchmark_suite_metadata(269)

    # Display summary table
    print("\n" + "=" * 80)
    print("DATASET SUMMARY TABLE - Basic Info")
    print("=" * 80)

    # Set display options for better viewing
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    summary_cols = [
        "dataset_id",
        "name",
        "n_instances",
        "n_features",
        "n_numeric_features",
        "n_categorical_features",
    ]

    print(metadata_df[summary_cols].to_string(index=False))

    # Filter datasets based on criteria
    print("\n" + "=" * 80)
    print("FILTERED DATASETS FOR BENCHMARKING")
    print("=" * 80)

    # Define filtering parameters
    FILTER_PARAMS = {
        "min_numeric_proportion": 0.5,  # At least 50% numeric features
        "max_missing_values": 0,  # No missing values
        "max_instances": 40000,  # Less than 40k instances
        "min_instances": 1000,  # At least 1k instances
        "min_features": 0,
        "max_features": None,
        "exclude_name_patterns": ["QSAR", "MIP-2016", "yprop", "topo", "wine"],
        "min_interpretability_score": 0.8,  # 80% interpretable features
    }

    print("\nFiltering criteria:")
    for param, value in FILTER_PARAMS.items():
        print(f"  {param}: {value}")

    filtered_df = filter_datasets(metadata_df, **FILTER_PARAMS)

    print(f"\nDatasets meeting criteria: {len(filtered_df)} / {len(metadata_df)}")
    print("\n" + "-" * 80)

    filter_cols = [
        "dataset_id",
        "name",
        "n_instances",
        "n_features",
        "n_numeric_features",
        "numeric_proportion",
        "interpretability_score",
    ]

    print(filtered_df[filter_cols].to_string(index=False))

    # Print filtered dataset IDs
    selected_ids = filtered_df["dataset_id"].tolist()
    print(f"IDs: {selected_ids}")

    # Download all filtered datasets
    downloaded = download_filtered_datasets(
        filtered_df,
        output_dir="paper/benchmarking_datasets/datasets",
        file_format="csv",
    )
