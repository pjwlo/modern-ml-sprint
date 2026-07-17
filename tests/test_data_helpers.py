from src.utils.data_helpers import train_test_split


def test_train_test_split_ratios():
    # Create a dummy dataset with 100 rows
    data = [[i, i * 2] for i in range(100)]

    # Split 80% train, 20% test
    train, test = train_test_split(data, test_size=0.2)

    # Assert exact counts
    assert len(train) == 80
    assert len(test) == 20
    assert len(train) + len(test) == 100


def test_train_test_split_no_overlap():
    data = [[i, i * 2] for i in range(50)]
    train, test = train_test_split(data, test_size=0.3)

    # Convert to sets of tuples to check for overlapping rows
    train_set = set(tuple(row) for row in train)
    test_set = set(tuple(row) for row in test)

    # Ensure the intersection is completely empty
    assert train_set.intersection(test_set) == set()
