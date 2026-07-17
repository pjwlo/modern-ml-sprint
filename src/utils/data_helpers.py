import random


def train_test_split(data, test_size=0.2):
    """
    Splits a dataset into training and testing sets.
    """
    # Create a shallow copy to avoid mutating the original input list
    shuffled_data = list(data)

    # Use a fixed seed so your splits are reproducible across runs
    random.seed(42)
    random.shuffle(shuffled_data)

    # Calculate the index boundary where the split happens
    split_idx = int(len(shuffled_data) * (1.0 - test_size))

    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]

    return train_data, test_data
