# Imputing variables across surveys

This document walks through a typical workflow for imputing variables from one survey to another using microimpute, using wealth imputation from the Survey of Consumer Finances (SCF) into the Current Population Survey (CPS) as the running example.

## Identifying receiver and donor datasets

Start by identifying your donor and receiver datasets. The donor dataset contains the variable you want to impute (here, the SCF contains wealth data). The receiver dataset will receive the imputed variable (here, the CPS, which lacks wealth data). Both datasets need predictor variables in common for the imputation to work. For example, both the SCF and CPS contain demographic and financial data that can help predict wealth.

```python
import pandas as pd
from microimpute.models import OLS, Matching, QRF

# Load donor dataset (SCF with wealth data)
scf_data = pd.read_csv("scf_data.csv")

# Load receiver dataset (CPS without wealth data)
cps_data = pd.read_csv("cps_data.csv")
```

## Cleaning and aligning variables

Before imputation, make sure both datasets have compatible variables. Identify common variables present in both datasets and standardize their formats, units, and categories so that Python can match them correctly. Handle missing values in common variables, and identify the target variables in the donor dataset that will inform the imputed values in the receiver. For details on preprocessing options, see the [Data preprocessing page](../imputation-benchmarking/preprocessing.md).

```python
# Identify common variables
common_variables = ['age', 'income', 'education', 'marital_status', 'region']

# Ensure variable formats match (example: education coding)
education_mapping = {
    1: "less_than_hs",
    2: "high_school",
    3: "some_college",
    4: "bachelor",
    5: "graduate"
}

# Apply standardization to both datasets
for dataset in [scf_data, cps_data]:
    dataset['education'] = dataset['education'].map(education_mapping)

    # Convert income to same units (thousands)
    if 'income' in dataset.columns:
        dataset['income'] = dataset['income'] / 1000

# Identify target variable in donor dataset
target_variable = ['networth']
```

## Performing imputation

Microimpute offers several methods for imputation across surveys, described in the [Models chapter](../models). The underlying approach differs by method, but the workflow stays the same. Here are two examples.

### Matching imputation

Matching finds similar observations in the donor dataset for each receiver observation and transfers the donor's target values. Fit on the donor dataset, then predict using the receiver dataset.

```python
# Set up matching imputer
matching = Matching()

# Train on donor dataset
fitted_matching = matching.fit(
    X_train=scf_data,
    predictors=common_variables,
    imputed_variables=target_variable,
)

# Impute target variable into receiver dataset
cps_with_wealth_matching = fitted_matching.predict(X_test=cps_data)
```

### Regression imputation (OLS)

OLS builds a linear regression model on the donor dataset and predicts wealth values for each combination of predictor values in the receiver. Again, fit on the donor and predict on the receiver.

```python
# Set up OLS imputer
ols = OLS()

# Train on donor dataset
fitted_ols = ols.fit(
    X_train=scf_data,
    predictors=common_variables,
    imputed_variables=target_variable,
)

# Impute target variable into receiver dataset
cps_with_wealth_ols = fitted_ols.predict(X_test=cps_data)
```

## Evaluating imputation quality

Evaluating imputation quality across surveys is challenging since the true values are unknown in the receiver dataset. Comparing the target variable's distribution in the donor to the imputed distribution in the receiver can reveal how well the imputation captures different parts of the distribution. Beyond mean or median accuracy, check performance at the tails using quantile loss. When comparing multiple methods, microimpute provides several metrics described in the [Metrics page](../imputation-benchmarking/metrics.md).

```python
from microimpute.comparisons import get_imputations
from microimpute.comparisons.metrics import compare_metrics

# Generate imputations from multiple models using cross-validation
method_imputations = get_imputations(
    model_classes=[QRF, OLS, Matching],
    X_train=train_data,
    X_test=test_data,
    predictors=common_variables,
    imputed_variables=target_variable,
)

# Compare quantile loss across methods
loss_comparison_df = compare_metrics(
    test_y=test_data[target_variable],
    method_imputations=method_imputations,
    imputed_variables=target_variable,
)
```

## Incorporating the imputed variable

Once you've chosen the best imputation method, incorporate the imputed variable into your receiver dataset for downstream analysis.

```python
# Choose the best imputation method (e.g., QRF)
final_imputed_dataset = cps_with_wealth_qrf

# Save the augmented dataset
final_imputed_dataset.to_csv("cps_with_imputed_wealth.csv", index=False)
```

## Key considerations

Model selection matters because different imputation methods have different strengths. QRF often performs better at capturing non-linear relationships, while Matching tends to preserve the original distributional properties of the data. Not all models handle categorical data: Matching can match any value regardless of type, but QuantReg does not support categorical imputation. OLS and QRF use logistic regression and random forest classification internally for categorical targets.

Variable selection is equally important. The common predictors should have strong explanatory power for the target variable. Because the ground truth is unknown in the receiver dataset, validation can involve simulation studies or comparison against known aggregate statistics.

For a complete worked example of the SCF-to-CPS net worth imputation pipeline, see the [autoimpute notebook](../autoimpute/autoimpute.ipynb). The [microimpute paper](https://github.com/PolicyEngine/microimpute/blob/main/paper/main.pdf) presents the full methodology and reports results from this imputation.
