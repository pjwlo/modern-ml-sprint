# import pytest
from src.pipeline import MLDataPipeline


def test_pipeline_fit_transform():
    # Simulated raw dataset: 4 rows with a numeric feature and a categorical feature
    raw_data = [
        {"age": 25, "city": "New York"},
        {"age": 30, "city": "Paris"},
        {"age": 22, "city": "New York"},
        {"age": 35, "city": "Tokyo"},
    ]

    pipeline = MLDataPipeline(categorical_features=["city"], numeric_features=["age"])

    # Process the data
    transformed_data, feature_names = pipeline.fit_transform(raw_data)

    # We expect 'age' + 3 unique cities sorted: ['Paris', 'Tokyo', 'New York']
    expected_features = ["age", "New York", "Paris", "Tokyo"]
    assert feature_names == expected_features

    # Check the structural transformations
    assert len(transformed_data) == 4
    assert transformed_data[0] == [25, 1, 0, 0]  # age=25, city="New York"
    assert transformed_data[1] == [30, 0, 1, 0]  # age=30, city="Paris"


0
