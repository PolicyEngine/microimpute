# Benchmarking different imputation methods

This chapter describes how microimpute lets you compare imputation methods using preprocessing, cross-validation, metric comparison, and evaluation tools.

The benchmarking functionality supports systematic comparison of multiple models on a common dataset. Cross-validation diagnoses overfitting and measures performance on held-out data where ground truth is available. Numerical imputation accuracy is assessed across quantiles via quantile loss; categorical imputation uses log loss. Visualizations show differences between methods, and predictor evaluation tools help inform variable selection when setting up the imputation task.
