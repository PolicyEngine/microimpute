"""Compatibility checks for public imports used by published examples."""


def test_legacy_scf_years_remain_importable() -> None:
    """Existing SCF examples retain their historical supported-year list."""
    from microimpute.config import VALID_YEARS

    assert VALID_YEARS == list(range(1989, 2023, 3))


def test_legacy_model_parameters_remain_importable() -> None:
    """The mapping stays importable for callers that still read it.

    QRF consumes its runtime defaults from this mapping. Other entries remain
    available for downstream callers; their models own runtime defaults.
    This compatibility check preserves imports and the expected mapping shape.
    """
    from microimpute.config import DEFAULT_MODEL_PARAMS

    assert set(DEFAULT_MODEL_PARAMS) == {"qrf", "quantreg", "ols", "matching", "mdn"}
    assert all(isinstance(v, dict) for v in DEFAULT_MODEL_PARAMS.values())
