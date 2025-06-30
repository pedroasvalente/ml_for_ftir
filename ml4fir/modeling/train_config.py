from sklearn.model_selection import KFold, StratifiedKFold

from ml4fir.config import random_seed

stratified_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=random_seed
)
regression_cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)


grid_search_args = {
    "scoring": "balanced_accuracy",
    "cv": 5,
}

grid_search_args_reg = {
    "scoring": "neg_mean_squared_error",
    "cv": regression_cv,
}

# WHY: why are the BayesSearchCV parameters different in each model? Like this you are not comparing
# the same search space for each model.
search_args = {
    "random_forest": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "mlp_classifier": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "decision_tree": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 100,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "xgboost": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
        },
    },
    "random_forest_regressor": {
        "GridSearchCV": grid_search_args_reg,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": regression_cv,
            "scoring": "neg_mean_squared_error",
        },
    },
    "mlp_regressor": {
        "GridSearchCV": grid_search_args_reg,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": regression_cv,
            "scoring": "neg_mean_squared_error",
        },
    },
    "decision_tree_regressor": {
        "GridSearchCV": grid_search_args_reg,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": regression_cv,
            "scoring": "neg_mean_squared_error",
        },
    },
    "xgboost_regressor": {
        "GridSearchCV": grid_search_args_reg,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": regression_cv,
            "scoring": "neg_mean_squared_error",
        },
    },
}


# WHY: why are the model_args_conf parameters different in each model? Like this you are not comparing the same model!
model_args_conf = {
    "mlp_classifier": {
        "GridSearchCV": {
            "max_iter": 3000,
            "learning_rate_init": 0.01,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
        "BayesSearchCV": {
            "validation_fraction": 0.1,
            "hidden_layer_sizes": (200,),
            "early_stopping": True,
        },
    },
}
