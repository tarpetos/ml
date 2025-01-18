import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


class LinearClassifier:
    def __init__(self):
        self.results = []

    def train_and_evaluate(self, X_train, X_val, y_train, y_val, params_list):
        for params in params_list:
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            accuracy = model.score(X_val, y_val) * 100
            self.results.append({
                "params": params,
                "accuracy": accuracy
            })

    def generate_report(self):
        columns = ["Regularization", "Max Iterations", "Solver Type", "Accuracy"]
        report_data = []

        for result in self.results:
            params = result["params"]
            report_data.append({
                "Regularization": params.get("C", "..."),
                "Max Iterations": params.get("max_iter", "..."),
                "Solver Type": params.get("solver", "..."),
                "Accuracy": f"{result['accuracy']:.2f}"
            })

        return pd.DataFrame(report_data)