from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH: Final[Path] = Path(__file__).parent.parent / "data"

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = StandardScaler()

    def prepare_data(self):
        df = pd.read_csv(self.file_path)
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna()

        features = ["Open", "High", "Low", "Close", "Volume"]
        x = df[features]
        y = df["Target"]

        x_scaled = self.scaler.fit_transform(x)
        x_train_val, x_test, y_train_val, y_test = train_test_split(x_scaled, y, test_size=0.1, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.22222, random_state=42)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def save_splits(self):
        x_train, x_val, x_test, y_train, y_val, y_test = self.prepare_data()

        splits = {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (x_test, y_test)
        }

        for split_name, (x, y) in splits.items():
            pd.DataFrame(x).to_csv(Path(DATA_PATH, f"x_{split_name}.csv"), index=False)
            pd.DataFrame(y).to_csv(Path(DATA_PATH, f"y_{split_name}.csv"), index=False)