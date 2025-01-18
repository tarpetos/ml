import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense


class NeuralNetworkModel:
    def __init__(self):
        self.results = []

    def create_model(self, params):
        model = Sequential([
            Dense(params["units_1"], activation="relu", input_shape=(5,)),
            Dense(params.get("units_2", 32), activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_and_evaluate(self, X_train, X_val, y_train, y_val, params_list):
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )

        for params in params_list:
            model = self.create_model(params)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[early_stopping],
                verbose=0
            )

            accuracy = model.evaluate(X_val, y_val, verbose=0)[1] * 100
            self.results.append({
                "params": params,
                "accuracy": accuracy
            })

    def generate_report(self):
        return pd.DataFrame([{
            "Parameter 1": result["params"].get("units_1", "..."),
            "Parameter 2": result["params"].get("units_2", "..."),
            "Parameter 3": "...",
            "Accuracy": f"{result['accuracy']:.2f}"
        } for result in self.results])
