from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from labs.data_preprocessor import DATA_PATH


class LinearClassifier:
    def __init__(self):
        self.results = []

    def train_and_evaluate(self, x_train, x_val, y_train, y_val, params_list):
        for params in params_list:
            model = LogisticRegression(**params)
            model.fit(x_train, y_train)
            accuracy = model.score(x_val, y_val) * 100
            self.results.append({"params": params, "accuracy": accuracy})

    def generate_report(self):
        report_data = []

        for result in self.results:
            params = result["params"]
            report_data.append(
                {
                "Regularization": params.get("C", "..."),
                "Max Iterations": params.get("max_iter", "..."),
                "Solver Type": params.get("solver", "..."),
                "Accuracy": f"{result['accuracy']:.2f}"
            }
            )

        pd.DataFrame(report_data).to_csv(Path(DATA_PATH, f"{Path(__file__).stem}.csv"), index=False)
