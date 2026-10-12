---
title: "microimpute: A Model-Agnostic Tool for Cross-Survey Imputation"
tags:
  - Python
  - imputation
  - statistical matching
  - survey microdata
  - quantile regression
  - microsimulation
authors:
  - name: Vahid Ahmadi
    orcid: 0009-0004-1093-6272
    affiliation: '1'
    corresponding: true
  - name: Max Ghenis
    orcid: 0000-0002-1335-8277
    affiliation: '1'
  - name: María Juaristi
    orcid: 0009-0007-4946-2248
    affiliation: '1'
  - name: Nikhil Woodruff
    orcid: 0009-0009-5004-4910
    affiliation: '1'
affiliations:
  - name: PolicyEngine, Washington, DC, United States
    index: '1'
date: 14 September 2026
bibliography: paper.bib
---

# Summary

`microimpute` imputes variables from one survey onto another and, more importantly, makes the choice of imputation method an empirical question rather than a convention. Policy microdata routinely lacks variables an analysis needs: a labour force survey records earnings but not wealth, a household survey records spending but not assets. The standard remedy is to borrow the variable from a richer donor survey conditional on characteristics both surveys observe. Many methods do this, they disagree, and the disagreement matters for the resulting estimates.

The package implements five approaches behind one `fit`/`predict` interface — statistical matching, ordinary least squares, quantile regression [@koenker1978regression], quantile regression forests [@meinshausen2006qrf], and mixture density networks [@bishop1994mdn] — and adds `autoimpute`, which cross-validates the available methods on the user's own data under five-fold cross-validation, optionally tuning hyperparameters, and selects by quantile loss for numerical targets or a categorical loss for categorical ones. Statistical matching wraps R's `StatMatch` through `rpy2` and mixture density networks require PyTorch; both are optional extras, and `autoimpute` compares whichever methods are installed. Ordinary least squares accepts survey weights directly and fits by weighted least squares; quantile regression forests accept a weight column, which is passed to the underlying forest. Quantile regression and mixture density networks raise an explicit error rather than silently returning an unweighted fit, so survey design is never discarded without the analyst knowing.

The design follows from an empirical finding rather than a preference. Benchmarking across six further datasets, alongside the wealth application, shows no method dominating across all of them: quantile regression forests win where relationships are nonlinear, while matching achieves the lowest mean rank overall because it draws from the donor pool directly and better preserves marginal distributions, with ordinary least squares and quantile regression occupying middle ranks [@juaristi2026microimpute]. With six benchmark datasets, the rank differences are not robust to the inclusion or exclusion of any single dataset. If method performance is dataset-specific, the useful tool is one that measures it.

# Statement of Need

Imputation choices are usually invisible in published analysis. A study reports a distributional result; the imputation method that produced the underlying variable is a sentence in an appendix, if it appears at all. Yet the choice can move headline numbers. In PolicyEngine's US model, imputing household wealth from the Survey of Consumer Finances onto the Current Population Survey is what makes asset tests bind at all: the model otherwise defaults countable resources to zero, so the baseline count of Supplemental Security Income recipients is overestimated by 167%, and a reform to the SSI asset limit cannot be simulated [@juaristi2026microimpute].

Analysts nonetheless tend to pick one method and keep it, because comparing methods is laborious. Each has a different API, different hyperparameters, and different output — a conditional mean from a regression, a donor record from matching, a predictive distribution from a forest. Building a like-for-like comparison means writing adapters and a cross-validation harness before any comparison happens, which is enough friction that the comparison usually is not done.

`microimpute` removes that friction. Because the methods are expressed as predictive distributions rather than point predictions, they are scored on the same footing with quantile loss across a common grid, and the comparison is a function call rather than a project. The package also makes the imputation reproducible: hyperparameter tuning, cross-validation, and selection run from a single entry point that records what was chosen.

# State of the Field

\renewcommand{\arraystretch}{1.5}

|  | `microimpute` | `scikit-learn` | `statsmodels` | R `mice` | R `StatMatch` |
|---|---|---|---|---|---|
| Multiple methods | 5 | 1 family | 1 family | Several | Matching |
| Automated selection | Yes | No | No | No | No |
| Quantile-based evaluation | Yes | No | No | No | No |
| Survey weights | Partly | No | No | No | Partly |
| Language | Python | Python | Python | R | R |

\renewcommand{\arraystretch}{1.0}

Three of `microimpute`'s five methods install with the package; statistical matching and mixture density networks are optional extras. `scikit-learn`'s `IterativeImputer` [@pedregosa2011scikit] and `statsmodels`' MICE [@seabold2010statsmodels] treat imputation as filling missing values within a dataset, which is a different problem from borrowing a variable across two surveys with no overlapping records. R's `mice` [@vanbuuren2011mice] is the reference implementation for multiple imputation by chained equations, and `StatMatch` [@dorazio2022statmatch] for statistical matching — the latter supporting donor weights in its random and rank hot-deck routines, though not in its distance-based nearest-neighbour hot deck — but neither compares across method families or selects between them, and using both means working in two idioms.

The gap `microimpute` fills is comparison. Its contribution is not a new estimator but a harness that makes existing estimators commensurable on a user's data, with an evaluation metric appropriate to distributional imputation.

# Software Design

Every model implements `fit(X_train, predictors, imputed_variables, weight_col=None)` and `predict(X_test, quantiles)`, returning quantiles of the conditional distribution. That uniformity is what makes the comparison possible: a regression and a donor-matching procedure are not obviously comparable until both are expressed as predictive distributions. Imputation is framed throughout as a donor-to-receiver problem: the donor survey observes both the predictors and the target variables, the receiver survey observes only the predictors, and the two share no records. Categorical predictors are encoded consistently across the two frames, so a model fitted on the donor can be applied to the receiver without the analyst reconciling schemas by hand. Optional numeric transformations — log, inverse hyperbolic sine and standardisation — are available for both frames.

![How `microimpute` works. A donor survey observing both the predictors and the targets, a receiver observing only the predictors, and a set of candidate methods feed a cross-validated comparison, which returns the imputed variables alongside the losses that chose the method.](architecture.png){width="62%"}


```python
from microimpute.comparisons import autoimpute

result = autoimpute(
    donor_data=scf,
    receiver_data=cps,
    predictors=["age", "income", "education"],
    imputed_variables=["net_worth"],
)
```

`autoimpute` runs each available method under five-fold cross-validation on the donor data, scores it by average quantile loss across a grid of quantiles, refits the winner on the full donor sample, and applies it to the receiver, returning the imputed values together with the comparison that justified them. Categorical and boolean targets are handled with log loss, and the target type is inferred rather than declared. Because the result carries the full per-method cross-validation table, the selection is auditable after the fact rather than buried in the run.

Alongside the imputers, the package provides diagnostics for the step that usually determines imputation quality more than the estimator does: the choice of predictors. `compute_predictor_correlations` reports Pearson and Spearman correlations among candidate predictors and mutual information between each predictor and each target, while `leave_one_out_analysis` and `progressive_predictor_inclusion` measure contribution by loss: the first by the degradation when a predictor is dropped, the second by building the predictor set up one addition at a time to find an ordering and a subset. A predictor set can then be defended rather than assumed. The zero-inflated wrapper composes a model for the probability of a zero with a model for the positive part, which matters for variables such as asset holdings where a large share of the population is at zero.

Results are inspectable rather than final: the package reports per-method losses so an analyst can see how close the decision was, and a companion web dashboard renders the comparison for exploration.

# Research Impact Statement

`microimpute` builds the imputed variables in the microdata underlying `policyengine` [@policyengine_py], the microsimulation model behind the analyses published at [policyengine.org](https://policyengine.org). Its SCF-to-CPS wealth imputation supplies the countable-resource inputs on which US asset-tested programme modelling depends, and its quantile regression forests impute variables into the UK microdata. It is also used in standalone studies, including a UK trade shock study and an analysis of a National Insurance contributions exemption.

The accompanying research paper documents the benchmarking exercise and the SSI application in full [@juaristi2026microimpute]; this paper describes the software.

# Acknowledgements

We thank Ben Ogorek for contributions to the package. Arnold Ventures [@arnold_ventures], NEO Philanthropy [@neo_philanthropy], the Gerald Huff Fund for Humanity, and the National Science Foundation (NSF POSE Phase I, Award 2518372) [@nsf_pose] funded this work in the US; the Nuffield Foundation has funded the UK work since September 2024 [@nuffield2024grant]. These funders had no involvement in the design, development, or content of this software or paper. All authors are employed by PolicyEngine and may benefit reputationally from the software's adoption; this relationship is disclosed as a potential conflict of interest.

# AI Usage Disclosure

The authors used generative AI tools, specifically Claude by Anthropic [@claude2026], to assist with code refactoring, test authoring, and drafting of this paper. Human authors reviewed, edited, and validated all AI-assisted outputs, and made all decisions regarding method implementations, evaluation design, and software architecture. The authors remain fully responsible for the accuracy, originality, and correctness of all submitted materials.

# References
