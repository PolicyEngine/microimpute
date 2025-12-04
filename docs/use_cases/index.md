# Imputing full variables across surveys

This document explains what the workflow for imputing variables from one survey to another using Microimpute may look like. We'll use the example of imputing wealth data from the Survey of Consumer Finances (SCF) into the Current Population Survey (CPS).

## Identifying receiver and donor datasets

The first step is to identify your donor and receiver datasets. The Donor dataset is that containing the variable you want to impute (e.g., SCF contains wealth data). The Receiver dataset will receive the imputed variable (e.g., CPS which originally did not contain wealth data but will after our imputation is completed). It is important for these two datasets to have predictor variables in common for the imputation to be succcesful. For example, both the SCF and CPS surveys contain demographic and financial data that may help us understand how wealth values may be distributed. 

```python
import pandas as pd
from microimpute.models import OLSImputer, MatchingImputer, QRFImputer

# Load donor dataset (SCF with wealth data)
scf_data = pd.read_csv("scf_data.csv")

# Load receiver dataset (CPS without wealth data)
cps_data = pd.read_csv("cps_data.csv")
```

## Cleaning and aligning variables

Before imputation, you need to ensure both datasets have compatible variables. Identify common variables present in both datasets
and standardize their variable formats, units, and categories so that Python can recognize they indeed represent the same the same data. Remember to also handle missing values in common variables. Lastly, identify the target variables in the donor dataset which will directly inform the values of the imputed variables in the receiver dataset. For details on data preprocessing options refer to the [Data preprocessing page](./preprocessing.md).

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

Microimpute offers several methods for imputation across surveys, which are described in the [Models Chapter](../models). The approach under the hood will differ based on the method chosen, although the workflow will remain constant. Let us see this for two different example methods.

### Matching imputation

Matching finds similar observations in the donor dataset for each observation in the receiver dataset and imputes the values for those receiver observations based on the values of the target value in the donor dataset. To do so it should be fitted on the donor dataset and predict using the receiver dataset. This will ensure the correct mapping of variables from one survey to the other. 

```python
# Set up matching imputer
matching_imputer = MatchingImputer(
    predictors=common_variables,
    imputed_variables=target_variable
)

# Train on donor dataset
matching_imputer.fit(scf_data)

# Impute target variable into receiver dataset
cps_data_with_wealth_matching = matching_imputer.predict(cps_data)
```

### Regression imputation (OLS)

OLS imputation builds a linear regression model using the donor dataset and applies it to the receiver dataset, predicting what wealth values may be for a specific combination of predictor variable values. To do so, again we need to make sure that we first fit the model on the donor dataset, while calling predict on the receiver dataset.

 ```python
# Set up OLS imputer
ols_imputer = OLSImputer(
    explanatory_variables=common_variables,
    target_variable=target_variable
)

# Train on donor dataset
ols_imputer.fit(scf_data)

# Impute target variable into receiver dataset
cps_data_with_wealth_ols = ols_imputer.impute(cps_data)
```

## Evaluating imputation quality

Evaluating imputation quality across surveys can be challenging since the true values aren't known in the receiver dataset. Comparing the distribution of the target variables in the donor dataset to the distribution of the variables we imputed in the receiver dataset may give us an understanding of the imputation quality for different sections of the distribution. We may want to pay particular attention to obtaining accurate prediction not only for mean or median values but also look at the performance at the distribution tails. This can be achieved computing the quantile loss supported by Microimpute. Additionally, if we have performed imputation accross multiple methods we may want to compare across them. Microimpute supports this through multiple easy-to-use metrics described in the [Metrics page](./metrics.md) file.

```python
# Ensure all imputations are in a dictionary mapping quantiles to dataframes containing imputed values
method_imputations = {
    [0.1]: pd.DataFrame
    [0.5]: pd.DataFrame
    ...
}

# Compare original wealth distribution with imputed wealth across methods
loss_comparison_df = compare_quantile_loss(Y_test, method_imputations)
```

## Incorporating the imputed variable

Once you've chosen the best imputation method, you may want to incorporate the imputed variable into your receiver dataset for future analysis.

```python
# Choose the best imputation method (e.g., QRF)
final_imputed_dataset = cps_data_with_wealth_qrf

# Save the augmented dataset
final_imputed_dataset.to_csv("cps_with_imputed_wealth.csv", index=False)
```

## Key considerations

Model selection plays a critical role in this workflow because different imputation methods have unique strengths. For example, a Quantile Regression Forest (QRF) often performs better when capturing complex relationships between variables, while a Matching approach may be more effective at preserving the original distributional properties of the data. Additionally, not all models can impute categorical data. For example, atching is able to match any value regardless of its data type, but QuantReg does not support categorical imputation. OLS and QRF will use logistic regression and random forest classification methods under the hood, respectively. Variable selection is equally important, since the common predictors across datasets should have strong power for explaining the target variable to ensure a reliable imputation. Because the ground truth is typically unknown in the receiver dataset, validation can involve simulation studies or comparing imputed values against known aggregate statistics. Finally, it is crucial to maintain documentation of the imputation process, from the choice of model to any adjustments made, so that the analysis remains transparent and reproducible for others. For the full pipeline details on SCF-to-CPS net worth imputation refer to the following [notebook](./scf_to_cps/imputing-from-scf-to-cps.md).
