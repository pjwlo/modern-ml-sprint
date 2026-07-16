def square_number(x):
    return x**2


def min_max_scale(data: list[float]) -> list[float]:
    """Scales a list of numbers to a range between 0 and 1."""
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)

    # Handle the edge case where all numbers are identical
    if max_val == min_val:
        return [0.0] * len(data)

    return [(x - min_val) / (max_val - min_val) for x in data]
