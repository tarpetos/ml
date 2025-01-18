import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()

    def prepare_data(self):
        df = pd.read_csv(self.file_path)
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()

        features = ["Open", "High", "Low", "Close", "Volume"]
        X = df[features]
        y = df["Target"]

        X_scaled = self.scaler.fit_transform(X)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.22222, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_splits(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()

        splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }

        for split_name, (X, y) in splits.items():
            pd.DataFrame(X).to_csv(f"X_{split_name}.csv", index=False)
            pd.DataFrame(y).to_csv(f"y_{split_name}.csv", index=False)