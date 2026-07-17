from src.utils.feature_engineering import one_hot_encode


class MLDataPipeline:

    def __init__(self, categorical_features=None, numeric_features=None):
        self.categorical_features = categorical_features or []
        self.numeric_features = numeric_features or []

    def fit_transform(self, data):
        """Transforms a list of dicts into a numerical matrix.

        Returns:
            transformed_data (list of lists): The processed numerical matrix.
            feature_names (list): The full list of feature column headers.
        """
        if not data:
            return [], []

        # 1. Gather all categorical columns to feed to our one-hot encoder
        # Right now we support one categorical feature for simplicity
        cat_col = self.categorical_features[0]
        categories = [row[cat_col] for row in data]

        # 2. Run our existing one-hot encoder utility
        encoded_matrix, encoded_feature_names = one_hot_encode(categories)

        # 3. Combine numeric features with the encoded categorical names
        # Feature order: numeric features first, then the encoded categories
        feature_names = self.numeric_features + encoded_feature_names

        # 4. Construct the final matrix row by row
        transformed_data = []
        for i, row in enumerate(data):
            # Extract numeric values
            num_values = [row[num_col] for num_col in self.numeric_features]
            # Combine numeric values with the corresponding encoded categorical row
            combined_row = num_values + encoded_matrix[i]
            transformed_data.append(combined_row)

        return transformed_data, feature_names
