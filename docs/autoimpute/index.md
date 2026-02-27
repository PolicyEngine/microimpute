# Autoimpute

The `autoimpute` function automates the entire method comparison, selection, and imputation pipeline in a single call.

The pipeline begins with input validation, then preprocesses the donor and receiver datasets for model training and evaluation. It supports numerical, categorical, and boolean variable types, selecting the appropriate method for each. At its core, `autoimpute` runs cross-validation on the donor data to evaluate multiple imputation methods. Each model is scored using quantile loss for numerical variables and log loss for categorical variables. The method with the lowest average loss (combining different metrics via a weighted-rank approach) across target variables is selected automatically. That model is then trained on the full donor dataset and applied to generate imputations for the receiver. The result is an `AutoImputeResult` object containing the imputations, the augmented receiver dataset, fitted models, and cross-validation results.
