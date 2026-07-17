def one_hot_encode(categories):
    """
    Encodes a list of categorical strings into a matrix of binary vectors.
    Returns:
        encoded_matrix (list of lists): The 1s and 0s representations.
        feature_names (list): The sorted unique categories matching the columns.
    """
    if not categories:
        return [], []

    # Extract unique categories and sort them to guarantee deterministic column order
    feature_names = sorted(list(set(categories)))

    # Map each category string to its index position in the sorted list
    category_to_idx = {category: idx for idx, category in enumerate(feature_names)}

    encoded_matrix = []
    for category in categories:
        # Create a row of zeros matching the number of unique features
        row = [0] * len(feature_names)
        # Flip the bit to 1 at the correct category index
        idx = category_to_idx[category]
        row[idx] = 1
        encoded_matrix.append(row)

    return encoded_matrix, feature_names
