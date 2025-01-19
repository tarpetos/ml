from pathlib import Path

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from labs.data_preprocessor import DATA_PATH


class NeuralNetworkModel:
    def __init__(self):
        self.results = []

    @staticmethod
    def create_model(params):
        model = Sequential([
            Dense(params["hidden_layer_1"], activation="relu", input_shape=(5,)),
            Dense(params.get("hidden_layer_2", 32), activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_and_evaluate(self, x_train, x_val, y_train, y_val, params_list):
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )

        for params in params_list:
            model = self.create_model(params)
            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=50,
                callbacks=[early_stopping],
                verbose=0
            )

            accuracy = model.evaluate(x_val, y_val, verbose=0)[1] * 100
            self.results.append({"params": params, "accuracy": accuracy})

    def generate_report(self):
        report_data = []

        for result in self.results:
            params = result["params"]
            report_data.append(
                {
                    "Hidden Layer 1": params.get("hidden_layer_1", "..."),
                    "Hidden Layer 2": params.get("hidden_layer_2", "..."),
                    "Learning Rate": params.get("learning_rate", "..."),
                    "Accuracy": f"{result['accuracy']:.2f}"
                }
            )

        pd.DataFrame(report_data).to_csv(Path(DATA_PATH, f"{Path(__file__).stem}.csv"), index=False)
