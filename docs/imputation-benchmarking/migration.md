# Migrating distributional imputation workflows

Declare target semantics before fitting or scoring. Numeric columns, including
integer counts and 0/1 integers, remain numeric. Strings, pandas categorical
columns and booleans use classification. Pass `target_types` to `fit`,
`autoimpute`, `cross_validate_model` and `compare_metrics` when the dtype does
not express the intended meaning. Supported declarations are `"numeric"`,
`"categorical"` and `"bool"`.

```python
import numpy as np
import pandas as pd
from microimpute.models import QRF
from microimpute.comparisons.metrics import compare_metrics

donor = pd.DataFrame({
    "age": np.arange(20., 60.),
    "count": np.tile([0, 1, 2, 3], 10),
    "status": np.tile([0, 1], 20),
})
targets = ["count", "status"]
types = {"count": "numeric", "status": "categorical"}
fitted = QRF(sequential=False, seed=42).fit(
    donor, ["age"], targets, target_types=types,
)
predictions = fitted.predict(
    donor[["age"]], quantiles=[0.1, 0.5, 0.9], return_probs=True,
)
scores = compare_metrics(
    donor, {"QRF": predictions}, targets, target_types=types,
)
```

Numeric regression can produce fractional counts or values outside a target's
observed support. Declaring a target categorical restricts predictions to fitted
classes, but changes the model and scoring metric. Choose the intended model
explicitly; rounding predictions does not establish a valid count distribution.

## QRF draws and quantiles

`QRF()` defaults to `sequential=True`. With multiple targets, each target uses
previous targets as predictors. Call `fitted.predict(receiver)` for a stochastic
joint donor draw. Repeated calls advance the fitted random streams.

Use `QRF(sequential=False)` for target-specific conditional marginal quantiles.
Each target then uses only the original predictors, and its seed depends on its
name rather than target ordering. A sequential fit with multiple targets rejects
explicit quantiles: chaining the same quantile across targets does not calculate
their marginal quantiles. Single-target fits support explicit quantiles.

The helper `microimpute.models.imputer.create_distributional_model(QRF, seed=42)`
constructs the independent mode used by comparison, cross-validation and
predictor-analysis workflows. Direct construction retains the sequential default.

Weighted QRF retains in-bag donor multiplicities and survey weights to calculate
the conditional empirical distribution. Retaining weighted leaves adds storage,
and prediction visits trees and rows in Python. Runtime and memory depend on
training size, forest settings and query size; this release does not establish a
universal performance improvement or a fixed memory bound.

## Categorical probabilities and scoring

Request `return_probs=True` when evaluating categorical targets. The result's
`"probabilities"` entry maps each target to a probability matrix and its ordered
`"classes"`. OLS, QRF and MDN include exact one-class point masses for constant
categorical and boolean targets. Numeric constants remain numeric forecasts.

`compare_metrics` aligns supplied class labels and probabilities. It raises
`ValueError` for missing or invalid categorical probabilities; hard labels cannot
substitute for probability forecasts. Comparison quantiles must be a nonempty
list of finite values in `[0, 1]`. Input validation also rejects duplicate column
names, overlapping predictors/targets, and incompatible predictor dtypes.

## Matching donor draws

`Matching.fit(...).predict(receiver)` returns a DataFrame containing one matched
donor draw per recipient. Quantile and probability requests raise
`NotImplementedError`. `autoimpute(..., impute_all=True, models=[OLS, Matching])`
includes Matching draws in the output while excluding Matching from
distributional ranking. Supply at least one distributional model for selection.
Distributional predictor analysis also rejects Matching at its public entry
points. Direct Matching hyperparameter tuning evaluates donor-draw errors.

Matching's seed controls native R adapter calls without changing the caller's
global R random stream. Both the default lazy adapter and directly imported
`microimpute.utils.statmatch_hotdeck.nnd_hotdeck_using_rpy2` support this behavior.
Custom adapters retain their existing argument contract. Weighted R matching
uses `RANDwNND.hotdeck`; constrained weighted matching remains unsupported.

## Existing fitted models and preprocessing

Historical pickled OLS, QRF and Matching results initialize missing random-state
attributes on loading. Historical QRF results preserve sequential conditioning.
Deserialization cannot reconstruct corrected survey-weight distributions from
an old fitted forest: refit from the original donor data and regenerate saved
predictions and comparison results to adopt the corrected algorithms.

For MDN, remove or regenerate old model-cache directories when upgrading; set
`force_retrain=True` when fitting to rebuild them. The MDN extra currently bounds
setuptools below 82 because its Lightning dependency imports `pkg_resources`.

Models returned by `autoimpute` retain transformations fitted on the donor data.
Pass raw receiver predictors to their `predict` method; the wrapper transforms
inputs and returns predictions in original units. Do not normalize the receiver
independently or transform it a second time. Cross-validation fits transformations
inside each training fold and scores predictions in original units.
