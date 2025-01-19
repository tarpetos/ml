from pathlib import Path
import numpy as np
import pandas as pd
from labs.data_preprocessor import DATA_PATH


class CustomLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.results = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def initialize_weights(self, n_features):
        limit = np.sqrt(2.0 / n_features)
        self.weights = np.random.normal(0, limit, n_features)
        self.bias = np.random.normal(0, limit, 1)[0]

    def train(self, x_train, y_train, learning_rate, epochs):
        n_samples, n_features = x_train.shape
        self.initialize_weights(n_features)

        x_train_np = x_train if isinstance(x_train, np.ndarray) else x_train.to_numpy()
        y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.to_numpy()

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(n_samples)
            x_shuffled = x_train_np[shuffle_idx]
            y_shuffled = y_train_np[shuffle_idx]

            linear_pred = np.dot(x_shuffled, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(x_shuffled.T, (predictions - y_shuffled))
            db = (1 / n_samples) * np.sum(predictions - y_shuffled)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, x):
        x_np = x if isinstance(x, np.ndarray) else x.to_numpy()
        linear_pred = np.dot(x_np, self.weights) + self.bias
        return self.sigmoid(linear_pred) >= 0.5

    def evaluate(self, x, y):
        predictions = self.predict(x)
        y_np = y if isinstance(y, np.ndarray) else y.to_numpy()
        return np.mean(predictions == y_np) * 100

    def train_and_evaluate(self, x_train, x_val, y_train, y_val, params_dict):
        for params in params_dict:
            self.train(x_train, y_train, params["learning_rate"], params["epochs"])
            accuracy = self.evaluate(x_val, y_val)
            self.results.append({"params": params, "accuracy": accuracy})

    def generate_report(self):
        report_data = []

        for result in self.results:
            params = result["params"]
            report_data.append(
                {
                    "Learning Rate": params.get("learning_rate", "..."),
                    "Epochs": params.get("epochs", "..."),
                    "Batch Size": params.get("batch_size", "..."),
                    "Accuracy": f"{result['accuracy']:.2f}"
                }
            )

        pd.DataFrame(report_data).to_csv(Path(DATA_PATH, f"{Path(__file__).stem}.csv"), index=False)