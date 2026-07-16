from src.utils.math_helpers import square_number
from src.utils.math_helpers import min_max_scale


def test_square_number():
    assert square_number(4) == 16
    assert square_number(0) == 0
    assert square_number(-2) == 4


def test_min_max_scale():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert min_max_scale(data) == expected


def test_min_max_scale_single_value():
    # If all values are the same, it should return a list of zeroes to prevent division by zero
    data = [10.0, 10.0, 10.0]
    assert min_max_scale(data) == [0.0, 0.0, 0.0]
