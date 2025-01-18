import numpy as np
import pandas as pd


class CustomLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.results = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X_train, y_train, learning_rate, epochs):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(epochs):
            linear_pred = np.dot(X_train, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X_train.T, (predictions - y_train))
            db = (1 / n_samples) * np.sum(predictions - y_train)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred) >= 0.5

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

    def generate_report(self):
        report_data = []

        for result in self.results:
            params = result["params"]
            report_data.append({
                "Learning Rate": params.get("learning_rate", "..."),
                "Epochs": params.get("epochs", "..."),
                "Batch Size": params.get("batch_size", "..."),
                "Accuracy": f"{result['accuracy']:.2f}"
            })

        return pd.DataFrame(report_data)