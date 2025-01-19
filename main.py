from itertools import product

from labs.data_preprocessor import DataPreprocessor, DATA_PATH
from labs.lab1_linear_classification import LinearClassifier
from labs.lab2_custom_linear_regression import CustomLinearRegression
from labs.lab3_gradient_boosting import GradientBoostingModel
from labs.lab4_neural_network import NeuralNetworkModel


def main() -> None:
    preprocessor = DataPreprocessor(DATA_PATH / "TESLA.csv")
    x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.prepare_data()
    preprocessor.save_splits()

    # LAB #1: Linear Classification
    linear_params_dict = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": [2500],
        "solver": ["lbfgs", "liblinear", "saga", "newton-cg"],
        "tol": [1e-4]
    }

    linear_params = [
        dict(zip(linear_params_dict.keys(), values))
        for values in product(*linear_params_dict.values())
    ]

    linear_clf = LinearClassifier()
    linear_clf.train_and_evaluate(x_train, x_val, y_train, y_val, linear_params)
    linear_clf.generate_report()

    # LAB #2: Custom Linear Regression
    custom_lr_params_dict = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "epochs": [100, 200, 500, 1000],
        "batch_size": [16, 32, 64, 128]
    }

    param_combinations = [
        dict(zip(custom_lr_params_dict.keys(), values))
        for values in product(*custom_lr_params_dict.values())
    ]

    custom_lr = CustomLinearRegression()
    custom_lr.train_and_evaluate(x_train, x_val, y_train, y_val, param_combinations)
    custom_lr.generate_report()

    # LAB #3: Gradient Boosting
    gb_params_dict = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2]
    }

    gb_params = [
        dict(zip(gb_params_dict.keys(), values))
        for values in product(*gb_params_dict.values())
    ]

    gb_model = GradientBoostingModel()
    gb_model.train_and_evaluate(x_train, x_val, y_train, y_val, gb_params)
    gb_model.generate_report()

    # LAB #4: Neural Networks
    nn_params_dict = {
        "hidden_layer_1": [32, 64, 128, 256],
        "hidden_layer_2": [16, 32, 64, 128],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1]
    }

    nn_params = [
        dict(zip(nn_params_dict.keys(), values))
        for values in product(*nn_params_dict.values())
    ]

    nn_model = NeuralNetworkModel()
    nn_model.train_and_evaluate(x_train, x_val, y_train, y_val, nn_params)
    nn_model.generate_report()


if __name__ == "__main__":
    main()
