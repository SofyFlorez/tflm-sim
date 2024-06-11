""" Example tests for the example module.
    This is only used to make sure the development environment is properly
    set up. To be removed once the project gets going.
"""

from tflm_sim.example import example_function


def test_example_function() -> None:
    expected_value = 42
    assert example_function() == expected_value
