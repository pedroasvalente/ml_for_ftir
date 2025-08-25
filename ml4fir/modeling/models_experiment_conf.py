from skopt.space import Categorical, Integer, Real

from ml4fir.modeling.models import (
    DecisionTreeConfig,
    DecisionTreeRegressorConfig,
    KerasMagiaRegressor,
    MLPConfig,
    MLPRegressorConfig,
    RandomForestConfig,
    RandomForestRegressorConfig,
    XGBoostConfig,
    XGBoostRegressorConfig,
)

random_forest_config = RandomForestConfig(
    n_estimators={
        "param_grid_args": [100, 200],
        "bayes_search_params": Integer(50, 300),
    },
    max_depth={
        "param_grid_args": [4, 8, 12],
        "bayes_search_params": Integer(3, 15),
    },
    max_features=["sqrt", "log2"],
    criterion=["gini", "entropy"],
    min_samples_split=Integer(2, 20),
    min_samples_leaf=Integer(1, 10),
    bootstrap=Categorical([True, False]),
)

mlp_config = MLPConfig(
    hidden_layer_sizes={
        "param_grid_args": [(50,), (100,), (50, 50)],
        "bayes_search_params": Categorical([(50,), (100,), (50, 50)]),
    },
    activation={
        "param_grid_args": ["tanh", "relu"],
        "bayes_search_params": Categorical(["relu", "tanh", "logistic"]),
    },
    solver={
        "param_grid_args": ["sgd", "adam"],
        "bayes_search_params": Categorical(["adam", "sgd"]),
    },
    alpha={
        "param_grid_args": [0.0001, 0.001],
        "bayes_search_params": Real(1e-6, 1e-1, prior="log-uniform"),
    },
    learning_rate={
        "param_grid_args": ["constant", "adaptive"],
        "bayes_search_params": Categorical(["constant", "adaptive"]),
    },
    learning_rate_init=Real(1e-4, 1e-2, prior="log-uniform"),
    max_iter=Integer(500, 3000),
)

decision_tree_config = DecisionTreeConfig(
    criterion={
        "param_grid_args": ["gini", "entropy"],
        "bayes_search_params": Categorical(["gini", "entropy"]),
    },
    splitter={
        "param_grid_args": ["best", "random"],
        "bayes_search_params": Categorical(["best", "random"]),
    },
    max_depth={
        "param_grid_args": [None, 10, 20],
        "bayes_search_params": Integer(1, 100),
    },
    min_samples_split={
        "param_grid_args": [2, 5, 10],
        "bayes_search_params": Integer(2, 50),
    },
    min_samples_leaf={
        "param_grid_args": [1, 2, 4],
        "bayes_search_params": Integer(1, 20),
    },
    max_features=Categorical([None, "sqrt", "log2"]),
)

xgboost_config = XGBoostConfig(
    n_estimators={
        "param_grid_args": [50, 100],
        "bayes_search_params": Integer(50, 500),
    },
    max_depth={
        "param_grid_args": [3, 6, 10],
        "bayes_search_params": Integer(1, 15),
    },
    learning_rate={
        "param_grid_args": [0.01, 0.1],
        "bayes_search_params": Real(0.01, 0.3, prior="log-uniform"),
    },
    subsample={
        "param_grid_args": [0.8, 1.0],
        "bayes_search_params": Real(0.5, 1.0),
    },
    colsample_bytree={
        "param_grid_args": [0.8, 1.0],
        "bayes_search_params": Real(0.5, 1.0),
    },
    min_child_weight=Integer(1, 10),
    gamma=Real(0, 5),
)


random_forest_reg_config = RandomForestRegressorConfig(
    n_estimators={
        "param_grid_args": [100, 200],
        "bayes_search_params": Integer(50, 300),
    },
    max_depth={
        "param_grid_args": [4, 8, 12],
        "bayes_search_params": Integer(3, 15),
    },
    max_features=["sqrt", "log2"],
    criterion=["squared_error", "absolute_error"],
    min_samples_split=Integer(2, 20),
    min_samples_leaf=Integer(1, 10),
    bootstrap=Categorical([True, False]),
)

mlp_reg_config = MLPRegressorConfig(
    hidden_layer_sizes={
        "param_grid_args": [(50,), (100,), (50, 50)],
        "bayes_search_params": Categorical([(50,), (100,), (50, 50)]),
    },
    activation={
        "param_grid_args": ["relu", "tanh"],
        "bayes_search_params": Categorical(["relu", "tanh", "logistic"]),
    },
    solver={
        "param_grid_args": ["adam", "sgd"],
        "bayes_search_params": Categorical(["adam", "sgd"]),
    },
    alpha={
        "param_grid_args": [0.0001, 0.001],
        "bayes_search_params": Real(1e-6, 1e-1, prior="log-uniform"),
    },
    learning_rate={
        "param_grid_args": ["constant", "adaptive"],
        "bayes_search_params": Categorical(["constant", "adaptive"]),
    },
    learning_rate_init=Real(1e-4, 1e-2, prior="log-uniform"),
    max_iter=Integer(500, 3000),
)

decision_tree_reg_config = DecisionTreeRegressorConfig(
    criterion={
        "param_grid_args": ["squared_error", "absolute_error"],
        "bayes_search_params": Categorical(["squared_error", "absolute_error"]),
    },
    splitter={
        "param_grid_args": ["best", "random"],
        "bayes_search_params": Categorical(["best", "random"]),
    },
    max_depth={
        "param_grid_args": [None, 10, 20],
        "bayes_search_params": Integer(1, 100),
    },
    min_samples_split={
        "param_grid_args": [2, 5, 10],
        "bayes_search_params": Integer(2, 50),
    },
    min_samples_leaf={
        "param_grid_args": [1, 2, 4],
        "bayes_search_params": Integer(1, 20),
    },
    max_features=Categorical([None, "sqrt", "log2"]),
)

xgboost_reg_config = XGBoostRegressorConfig(
    n_estimators={
        "param_grid_args": [50, 100],
        "bayes_search_params": Integer(50, 500),
    },
    max_depth={
        "param_grid_args": [3, 6, 10],
        "bayes_search_params": Integer(1, 15),
    },
    learning_rate={
        "param_grid_args": [0.01, 0.1],
        "bayes_search_params": Real(0.01, 0.3, prior="log-uniform"),
    },
    subsample={
        "param_grid_args": [0.8, 1.0],
        "bayes_search_params": Real(0.5, 1.0),
    },
    colsample_bytree={
        "param_grid_args": [0.8, 1.0],
        "bayes_search_params": Real(0.5, 1.0),
    },
    min_child_weight=Integer(1, 10),
    gamma=Real(0, 5),
    # objective is set internally in the config class to "reg:squarederror"
)


# cnn_config = KerasMagiaClassifier(
#     name="cnn",
#     desc_name="Convolutional Neural Network",
#     model_arch="CNN",
#     model_kwargs={
#         "filters": [16, 32, 64],
#         "kernel_size": [3, 5],
#         "pool_size": [2, 3],
#         "dropout": [0.3, 0.5],
#         "num_sequences": [1, 2, 3],
#     },
#     compile_kwargs={
#         "optimizer": ["adam", "rmsprop"],
#         "loss": ["categorical_crossentropy", "binary_crossentropy"],
#         "metrics": ["accuracy"],
#     },
#     fit_kwargs={
#         "epochs": [50, 100],
#     },
# )

# fcnn_config = KerasMagiaClassifier(
#     name="fcnn",
#     desc_name="Fully Connected Neural Network",
#     model_arch="FCNN",
#     model_kwargs={
#         "input_dim": [16, 32, 64],
#         "activation": ["relu"],
#         "division_base_power": [2, 3],
#         "num_sequences": [1, 2, 3],
#     },
#     compile_kwargs={
#         "optimizer": "adam",
#         "loss": "categorical_crossentropy",
#         "metrics": ["accuracy"],
#     },
#     fit_kwargs={
#         "epochs": [50, 100],
#     },
# )

# unet_config = KerasMagiaClassifier(
#     name="unet",
#     desc_name="UNet",
#     model_arch="UNET",
#     model_kwargs={
#         "input_shape": [(64, 64, 1), (128, 128, 1)],
#         "num_classes": [2, 3],
#         "filters": [[32, 64, 128], [64, 128, 256]],
#         "dropout": [0.3, 0.5],
#     },
#     compile_kwargs={
#         "optimizer": ["adam", "rmsprop"],
#         "loss": ["categorical_crossentropy", "binary_crossentropy"],
#         "metrics": ["accuracy"],
#     },
#     fit_kwargs={
#         "epochs": [50, 100],
#     },
# )


cnn_regressor_config = KerasMagiaRegressor(
    name="cnn_regressor",
    desc_name="CNN Regressor",
    model_arch="CNN",
    model_kwargs={
        "filters": [None, 16, 32, 64],
        "kernel_size": [3],
        "pool_size": [3],
        "num_sequences": [1, 2, 4],
        "interpretation_filters": [4, 8, 16],
        "double_interpretation": [True],
        "sklearn_wrapper":[True],
        "activation_end": ["relu"],

    # "filters": [16],
    # "kernel_size": [3],
    # "pool_size": [2],
    # "num_sequences": [1],
    # "interpretation_filters": [4],
    # "double_interpretation": [True],
    # "sklearn_wrapper": [True],
    # "activation_end": ["linear"],

    },
    compile_kwargs={
        "optimizer": ["adam"],
        "loss": ["mse"],
        "metrics": [["mae", "mape"]],
    },
    fit_kwargs={
        "epochs": [300],
        "verbose": [0],
    },
)

fcnn_regressor_config = KerasMagiaRegressor(
    name="fcnn_regressor",
    desc_name="FCNN Regressor",
    model_arch="FCNN",
    model_kwargs={
        "filters": [64, 32, None, 16],
        "num_sequences": [4, 1, 2],
        "interpretation_filters": [8, 16,4],
        "double_interpretation": [True],
        "activation_end": ["relu"],
        "division_per_dim":[False,True]
    },
    compile_kwargs={
        "optimizer": ["adam"],
        "loss": ["mse"],
        "metrics": [["mae", "mape"]],
    },
    fit_kwargs={
        "epochs": [300],
        "verbose": [0],
    },
)

unet_regressor_config = KerasMagiaRegressor(
    name="unet_regressor",
    desc_name="UNet Regressor",
    model_arch="UNET",
    model_kwargs={
        "n_filters": [16,32, 64],
        "classes_method":["dense"],
        "activation_end": ["relu"],
        "sklearn_wrapper":[True],
        "interpretation_filters": [4, 8, 16],
        "double_interpretation": [True],
        # "n_filters": [16],
        # "classes_method": ["dense"],
        # "activation_end": ["linear"],
        # "interpretation_filters": [4],
        # "sklearn_wrapper": [True],
        # "double_interpretation": [True],

    },
    compile_kwargs={
        "optimizer": ["adam"],
        "loss": ["mse"],
        "metrics": [["mae", "mape"]],
    },
    fit_kwargs={
        "epochs": [300],
        "verbose": [0],
    },
)


resnet_regressor_config = KerasMagiaRegressor(
    name="resnet_regressor",
    desc_name="ResNET Regressor",
    model_arch="ResNET",
    model_kwargs={
        "n_filters": [16,32, 64],
        "classes_method":["dense"],
        "activation_end": ["relu"],
        "sklearn_wrapper":[True],
        "interpretation_filters": [4, 8, 16],
        "double_interpretation": [True],
        # "n_filters": [16],
        # "classes_method":["dense", "fcnn", "conv"],
        # "activation_end": ["linear"],
        # "interpretation_filters": [4],
        # "sklearn_wrapper": [True],
        # "double_interpretation": [True],

    },
    compile_kwargs={
        "optimizer": ["adam"],
        "loss": ["mse"],
        "metrics": [["mae", "mape"]],
    },
    fit_kwargs={
        "epochs": [300],
        "verbose": [0],
    },
)


attresnet_regressor_config = KerasMagiaRegressor(
    name="attresnet_regressor",
    desc_name="AttResUNet Regressor",
    model_arch="AttResUNet",
    model_kwargs={
        "n_filters": [16,32, 64],
        "classes_method":["dense"],
        "activation_end": ["relu"],
        "interpretation_filters": [4, 8, 16],
        "sklearn_wrapper":[True],
        "double_interpretation": [True],
        # "n_filters": [16],
        # "classes_method": ["dense"],
        # "activation_end": ["linear"],
        # "interpretation_filters": [4],
        # "sklearn_wrapper": [True],
        # "double_interpretation": [True],

    },
    compile_kwargs={
        "optimizer": ["adam"],
        "loss": ["mse"],
        "metrics": [["mae", "mape"]],
    },
    fit_kwargs={
        "epochs": [300],
        "verbose": [0],
    },
)


resunet_regressor_config = KerasMagiaRegressor(
    name="resunet_regressor",
    desc_name="ResUNet Regressor",
    model_arch="ResUNet",
    model_kwargs={
        "n_filters": [16,32, 64],
        "classes_method":["dense"],
        "activation_end": ["relu"],
        "sklearn_wrapper":[True],
        "interpretation_filters": [4, 8, 16],
        "double_interpretation": [True],
    # "n_filters": [16],
    # "classes_method": ["dense"],
    # "activation_end": ["linear"],
    # "interpretation_filters": [4],
    # "sklearn_wrapper": [True],
    # "double_interpretation": [True],

    },
    compile_kwargs={
        "optimizer": ["adam"],
        "loss": ["mse"],
        "metrics": [["mae", "mape"]],
    },
    fit_kwargs={
        "epochs": [300],
        "verbose": [0],
    },
)

models_experiment = {
    "random_forest": random_forest_config,
    "mlp_classifier": mlp_config,
    "decision_tree": decision_tree_config,
    "xgboost": xgboost_config,
    "random_forest_regressor": random_forest_reg_config,
    "mlp_regressor": mlp_reg_config,
    "decision_tree_regressor": decision_tree_reg_config,
    "xgboost_regressor": xgboost_reg_config,
    # "cnn": cnn_config,
    # "fcnn": fcnn_config,
    # "unet": unet_config,
    "cnn_regressor": cnn_regressor_config,
    "fcnn_regressor": fcnn_regressor_config,
    "unet_regressor": unet_regressor_config,
    "resnet_regressor": resnet_regressor_config,
    "attresnet_regressor": attresnet_regressor_config,
    "resunet_regressor": resunet_regressor_config,
}
