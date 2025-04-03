from zz_config_module import random_seed, stratified_cv

grid_search_args = {
    "scoring": "balanced_accuracy",
    "cv": 5,
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
            "random_state": random_seed,
        },
    },
    "mlp_classifier": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 10,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
            "random_state": random_seed,
        },
    },
    "decision_tree": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "n_points": 100,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
            "random_state": random_seed,
        },
    },
    "xboost": {
        "GridSearchCV": grid_search_args,
        "BayesSearchCV": {
            "n_iter": 50,
            "cv": stratified_cv,
            "scoring": "balanced_accuracy",
            "random_state": random_seed,
        },
    },
}

# WHY: why are the model_args_conf parameters different in each model? Like this you are not comparing the same model!
model_args_conf = {
    "mlp_classifier": {
        "GridSearchCV": {
            "max_iter": 3000,
            "random_state": "random_seed",
            "learning_rate_init": 0.01,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
        "BayesSearchCV": {
            "random_state": "random_seed",
            "validation_fraction": 0.1,
            "hidden_layer_sizes": (200,),
            "early_stopping": True,
        },
    },
}
