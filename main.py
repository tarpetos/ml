from labs.data_preprocessor import DataPreprocessor
from labs.lab1_linear_classification import LinearClassifier
from labs.lab2_custom_linear_regression import CustomLinearRegression
from labs.lab3_gradient_boosting import GradientBoostingModel
from labs.lab4_neural_network import NeuralNetworkModel


def main() -> None:
    # # Data Preprocessing
    preprocessor = DataPreprocessor("data/TESLA.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data()
    preprocessor.save_splits()

    # # Task 1: Linear Classification
    # linear_params = [
    #     {"C": 0.1, "max_iter": 2500, "solver": "lbfgs", "tol": 1e-4},
    #     {"C": 1.0, "max_iter": 2500, "solver": "liblinear", "tol": 1e-4},
    #     {"C": 10.0, "max_iter": 2500, "solver": "saga", "tol": 1e-4}
    # ]
    # linear_clf = LinearClassifier()
    # linear_clf.train_and_evaluate(X_train, X_val, y_train, y_val, linear_params)
    # linear_report = linear_clf.generate_report()
    # print("Linear Classification Report:")
    # print(linear_report)
    #
    # # Task 2: Custom Linear Regression
    # custom_lr = CustomLinearRegression()
    # param_combinations = [
    #     {"learning_rate": 0.001, "epochs": 100, "batch_size": 32},
    #     {"learning_rate": 0.01, "epochs": 500, "batch_size": 64},
    #     {"learning_rate": 0.1, "epochs": 1000, "batch_size": 128}
    # ]
    #
    # for params in param_combinations:
    #     custom_lr.train(X_train, y_train, params["learning_rate"], params["epochs"])
    #     accuracy = custom_lr.evaluate(X_val, y_val)
    #     custom_lr.results.append({
    #         "params": params,
    #         "accuracy": accuracy
    #     })
    #
    # custom_report = custom_lr.generate_report()
    # print("\nCustom Linear Regression Report:")
    # print(custom_report)
    #
    # # Task 3: Gradient Boosting
    # gb_params = [
    #     {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    #     {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    #     {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.01}
    # ]
    # gb_model = GradientBoostingModel()
    # gb_model.train_and_evaluate(X_train, X_val, y_train, y_val, gb_params)
    # gb_report = gb_model.generate_report()
    # print("\nGradient Boosting Report:")
    # print(gb_report)
    # Task 4: Neural Networks
    nn_params = [
        {"units_1": 64, "units_2": 32},
        {"units_1": 128, "units_2": 64},
        {"units_1": 256, "units_2": 128}
    ]
    nn_model = NeuralNetworkModel()
    nn_model.train_and_evaluate(X_train, X_val, y_train, y_val, nn_params)
    nn_report = nn_model.generate_report()
    print("\nNeural Network Report:")
    print(nn_report)

if __name__ == "__main__":
    main()
