from src.utils.math_helpers import square_number

def test_square_number():
    assert square_number(4) == 16
    assert square_number(0) == 0
    assert square_number(-2) == 4
