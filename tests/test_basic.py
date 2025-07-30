"""Basic tests to verify CI setup."""


def test_import() -> None:
    """Test that the package can be imported."""
    import microimpute

    assert microimpute is not None
