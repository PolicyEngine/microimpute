---
title: "microimpute: a model-agnostic tool for cross-survey imputation"
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
  - name: María Juaristi
    affiliation: '1'
  - name: Max Ghenis
    orcid: 0000-0002-1335-8277
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

The package implements five approaches behind one `fit`/`predict` interface — statistical matching, ordinary least squares, quantile regression [@koenker1978regression], quantile regression forests [@meinshausen2006qrf], and mixture density networks [@bishop1994mdn] — and adds `autoimpute`, which tunes hyperparameters, cross-validates every method on the user's own data, and selects by quantile loss for numerical targets or log loss for categorical ones. Sample weights are supported throughout, so survey design is not lost at the imputation step.

The design follows from an empirical finding rather than a preference. Benchmarking across seven domains shows that no method dominates: quantile regression forests win where relationships are nonlinear, matching better preserves marginal distributions because it draws from the donor pool directly, and ordinary least squares remains competitive when relationships are close to linear [@juaristi2026microimpute]. If method performance is dataset-specific, the useful tool is one that measures it.

# Statement of Need

Imputation choices are usually invisible in published analysis. A study reports a distributional result; the imputation method that produced the underlying variable is a sentence in an appendix, if it appears at all. Yet the choice can move headline numbers. In PolicyEngine's US model, imputing household wealth from the Survey of Consumer Finances onto the Current Population Survey is what makes asset-tested programmes modellable at all: without imputed wealth the baseline count of Supplemental Security Income recipients is overestimated by 167%, and a reform to the SSI asset limit cannot be simulated [@juaristi2026microimpute].

Analysts nonetheless tend to pick one method and keep it, because comparing methods is laborious. Each has a different API, different hyperparameters, and different output — a conditional mean from a regression, a donor record from matching, a predictive distribution from a forest. Building a like-for-like comparison means writing adapters and a cross-validation harness before any comparison happens, which is enough friction that the comparison usually is not done.

`microimpute` removes that friction. Because every method returns quantiles of the conditional distribution rather than a point prediction, they can be scored on the same footing with quantile loss, and the comparison is a function call rather than a project. The package also makes the imputation reproducible: hyperparameter tuning, cross-validation, and selection run from a single entry point that records what was chosen.

# State of the Field

| Tool | Multiple methods | Automated selection | Quantile-based evaluation | Survey weights | Language |
|---|---|---|---|---|---|
| `microimpute` | 5 | Yes | Yes | Yes | Python |
| `scikit-learn` `IterativeImputer` [@pedregosa2011scikit] | 1 family | No | No | No | Python |
| `statsmodels` MICE [@seabold2010statsmodels] | 1 | No | No | Partly | Python |
| R `mice` [@vanbuuren2011mice] | Several | No | No | Partly | R |
| R `StatMatch` [@dorazio2022statmatch] | Matching | No | No | Yes | R |

`scikit-learn` and `statsmodels` treat imputation as filling missing values within a dataset, which is a different problem from borrowing a variable across two surveys with no overlapping records. R's `mice` is the reference implementation for multiple imputation by chained equations, and `StatMatch` for statistical matching, but neither compares across method families or selects between them, and using both means working in two idioms.

The gap `microimpute` fills is comparison. Its contribution is not a new estimator but a harness that makes existing estimators commensurable on a user's data, with an evaluation metric appropriate to distributional imputation.

# Software Design

Every model implements `fit(X, y, weights=None)` and `predict(X, quantiles)`, returning quantiles of the conditional distribution. That uniformity is what makes the comparison possible: a regression and a donor-matching procedure are not obviously comparable until both are expressed as predictive distributions.

```python
from microimpute.comparisons import autoimpute

result = autoimpute(
    donor_data=scf,
    receiver_data=cps,
    predictors=["age", "income", "education"],
    imputed_variables=["net_worth"],
)
```

`autoimpute` runs each method under cross-validation, scores it by average quantile loss across a grid of quantiles, and returns imputed values from the winner along with the comparison that justified it. Categorical and boolean targets are handled with log loss. The zero-inflated wrapper composes a model for the probability of a zero with a model for the positive part, which matters for variables such as asset holdings where a large share of the population is at zero.

Results are inspectable rather than final: the package reports per-method losses so an analyst can see how close the decision was, and a dashboard renders the comparison for exploration.

# Research Impact Statement

`microimpute` is used in `policyengine-uk-data`, which builds the microdata behind PolicyEngine's UK analyses, and in standalone studies including a UK trade shock study, a national insurance exemption analysis, and UK public services imputation. Its SCF-to-CPS wealth imputation is a dependency of PolicyEngine's US asset-tested programme modelling.

The accompanying research paper documents the benchmarking exercise and the SSI application in full [@juaristi2026microimpute]; this paper describes the software.

# Acknowledgements

We thank Ben Ogorek for contributions to the package. Arnold Ventures [@arnold_ventures], NEO Philanthropy [@neo_philanthropy], the Gerald Huff Fund for Humanity, and the National Science Foundation (NSF POSE Phase I, Award 2518372) [@nsf_pose] funded this work in the US; the Nuffield Foundation has funded the UK work since September 2024 [@nuffield2024grant]. These funders had no involvement in the design, development, or content of this software or paper. All authors are employed by PolicyEngine and may benefit reputationally from the software's adoption; this relationship is disclosed as a potential conflict of interest.

# AI Usage Disclosure

The authors used generative AI tools, specifically Claude by Anthropic [@claude2026], to assist with code refactoring, test authoring, and drafting of this paper. Human authors reviewed, edited, and validated all AI-assisted outputs, and made all decisions regarding method implementations, evaluation design, and software architecture. The authors remain fully responsible for the accuracy, originality, and correctness of all submitted materials.

# References
