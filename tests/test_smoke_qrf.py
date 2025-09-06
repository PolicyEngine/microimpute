"""
Minimal smoke test for Python 3.12 compatibility.
Tests only the core QRF functionality that PolicyEngine actually uses.
"""

import pandas as pd
import numpy as np
from microimpute.models.qrf import QRF


def test_qrf_basic_usage():
    """Test basic QRF usage as PolicyEngine uses it."""
    # Create simple test data
    np.random.seed(42)
    n_samples = 100

    X_train = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples),
            "income": np.random.randint(10000, 100000, n_samples),
            "household_size": np.random.randint(1, 6, n_samples),
        }
    )
    X_train["benefits"] = X_train["income"] * 0.1 + np.random.normal(
        0, 1000, n_samples
    )

    predictors = ["age", "income", "household_size"]
    imputed_variables = ["benefits"]

    # Test QRF instantiation with parameters PolicyEngine uses
    qrf = QRF(
        log_level="ERROR",  # Suppress logs for smoke test
        memory_efficient=True,
        batch_size=10,
        cleanup_interval=5,
    )

    # Test fit
    fitted_model = qrf.fit(
        X_train=X_train,
        predictors=predictors,
        imputed_variables=imputed_variables,
        n_jobs=1,  # Single thread as PolicyEngine uses
    )

    # Test predict
    X_test = X_train.iloc[:10].copy()
    predictions = fitted_model.predict(X_test=X_test)

    # Basic assertions
    assert "benefits" in predictions, "Should have predictions for 'benefits'"
    assert len(predictions["benefits"]) == len(
        X_test
    ), "Should have predictions for all test samples"
    assert (
        not predictions["benefits"].isna().any()
    ), "Should not have NaN predictions"

    print("✓ QRF smoke test passed")


if __name__ == "__main__":
    test_qrf_basic_usage()
