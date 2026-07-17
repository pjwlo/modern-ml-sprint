from src.utils.feature_engineering import one_hot_encode


def test_one_hot_encode_basic():
    categories = ["red", "blue", "green", "blue"]

    # We expect back the encoded data matrix and the unique feature columns order
    encoded_matrix, feature_names = one_hot_encode(categories)

    # Check that columns are sorted/determined cleanly
    assert feature_names == ["blue", "green", "red"]

    # Check the exact binary mappings for each row
    assert encoded_matrix[0] == [0, 0, 1]  # "red"
    assert encoded_matrix[1] == [1, 0, 0]  # "blue"
    assert encoded_matrix[2] == [0, 1, 0]  # "green"
    assert encoded_matrix[3] == [1, 0, 0]  # "blue"


def test_one_hot_encode_empty():
    encoded_matrix, feature_names = one_hot_encode([])
    assert encoded_matrix == []
    assert feature_names == []
