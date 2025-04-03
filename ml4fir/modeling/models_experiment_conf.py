from skopt.space import Categorical, Integer, Real

from ml4fir.modeling.models import (
                                    DecisionTreeConfig,
                                    MLPConfig,
                                    RandomForestConfig,
                                    XGBoostConfig,
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

print("Random Forest Config:", random_forest_config)
print("MLP Config:", mlp_config)
print("Decision Tree Config:", decision_tree_config)
print("XGBoost Config:", xgboost_config)
print("XGBoost Config:", xgboost_config)